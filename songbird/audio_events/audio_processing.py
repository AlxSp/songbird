import torchaudio
import torch
import librosa
#import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
import numba as nb
import pandas as pd
import math
import os
import json
import time
import argparse
import os
import csv
import shutil
import hdbscan
import multiprocessing
from collections import namedtuple

import warnings
warnings.filterwarnings('ignore') #filter warnings to suppress; warnings.warn("PySoundFile failed. Trying audioread instead.")

this_file_dir = os.path.dirname(os.path.abspath(__file__)) #absolute path to this file's directory
project_base_dir = os.path.dirname(os.path.dirname(this_file_dir)) #path to base dir of project
data_dir = os.path.join(project_base_dir, 'data')  #path to data_dir
reports_dir = os.path.join(project_base_dir, 'reports')

raw_sample_dir = os.path.join(data_dir, 'raw') #path pointing to samples
audio_events_dir = os.path.join(data_dir, 'audio_events') #path pointing to samples
samples_metadata_file_path = os.path.join(data_dir, 'data_dictionary', 'samples_metadata.json')
download_species_sample_path = os.path.join(data_dir, 'data_dictionary', 'download_species_sample_info.json')
plots_path = os.path.join(reports_dir, 'audio_event_plots')


######################################################################################################################
######################################################################################################################


AudioConversionParameters = namedtuple('AudioConversionParameters', ['sample_rate', 'window_size', 'step_size', 'max_frequency', 'min_frequency'])
EventDetectionParameters = namedtuple('EventDetectionParameters', ['mean_lag_window_size', 'std_lag_window_size', 'mean_influence', 'std_influence', 'threshold'])
ClusteringParameters = namedtuple('ClusteringParameters', ['min_cluster_size'])
EventProcessingParameters = namedtuple('EventProcessingParameters', ['event_distance_max', 'event_freq_differnce_max', 'event_length_min', 'start_buffer_len', 'end_buffer_len'])
AdditionalParameters = namedtuple('AdditionalParameters', 'generate_process_plots')

def getZScoreParameters():
    return AudioConversionParameters, EventDetectionParameters, ClusteringParameters, EventProcessingParameters, AdditionalParameters

CustomAudioConversionParameters = namedtuple('CustomAudioConversionParameters', [ 'sample_rate', 'window_size', 'step_size', 'max_frequency', 'min_frequency'])
CustomAudioProcessingParameters = namedtuple('CustomAudioProcessingParameters', ['fn_process_spectrogram', 'relevant_freq_range'])
CustomEventDetectionParameters = namedtuple('CustomEventDetectionParameters', ['fn_detect_peaks_in_spectrogram', 'mean_lag_window_size', 'std_lag_window_size', 'mean_influence', 'std_influence', 'threshold'])
CustomClusteringParameters = namedtuple('CustomClusteringParameters', ['fn_cluster_audio_events', 'min_cluster_size'])
CustomEventProcessingParameters = namedtuple('CustomEventProcessingParameters', ['event_distance_max', 'event_freq_differnce_max', 'event_length_min', 'start_buffer_len', 'end_buffer_len'])
AdditionalParameters = namedtuple('AdditionalParameters', 'generate_process_plots')

def getCustomParameters():
    return CustomAudioConversionParameters, CustomAudioProcessingParameters, CustomEventDetectionParameters, CustomClusteringParameters, CustomEventProcessingParameters, AdditionalParameters

######################################################################################################################
# Utility ############################################################################################################

def empty_or_create_dir(dir_path):
    '''
    creates or empties the directory pointed to by dir_path
    '''
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {}. Exceptions: {}'.format(file_path, e)) 
    else:
        os.makedirs(dir_path)


def get_all_samples_ids(dir_path):
    return [os.path.splitext(f)[0] for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

def load_audio_sample(sample_id, sample_rate, sample_dir = None):
    '''
    loads an audio sample by it's id (file name). Returns the audio and the the sample rate
    '''
    sample_dir = raw_sample_dir if sample_dir is None else sample_dir
    audio_file_path = os.path.join(sample_dir, str(sample_id) + '.mp3')
    if not os.path.isfile(audio_file_path): #check if file exists
        raise Exception(f"Sample with id '{sample_id}' does not exist! Available files can be found in the {sample_dir} directory")
    #sample audio at 44.1 khz and get the time series as a numpy array
    time_series, sample_rate = librosa.load(audio_file_path, sr = sample_rate)
    #trim empty start and end padding from time series
    time_series, _ = librosa.effects.trim(time_series)

    return time_series, sample_rate

def load_audio_events(sample_id):
    file_path = os.path.join(audio_events_dir, sample_id + '.csv')
    with open(file_path, mode='r') as file:
        df = pd.read_csv(file)
    return df.to_dict('records')

# Utility ############################################################################################################
######################################################################################################################

######################################################################################################################
# Time Series to Spectogram ##########################################################################################

def create_spectrogram(sample, window_size, step_size, sample_rate, reference_power = None):
    #create time series frames
    time_series_frames = get_even_time_series_frames(sample, window_size, step_size)
    #apply taper function to each frame
    apply_hanning_window(time_series_frames)
    #calculate the power spectrum
    power_frames = calculate_power_spectrum(time_series_frames, window_size)
    #get the frequency bins from for the window size
    rfft_bin_freq = get_frequency_bins(window_size, sample_rate)
    #convert power spectrogram to decibel units
    #db_frames =  librosa.power_to_db(power_frames, ref=1)
    db_frames = torchaudio.transforms.AmplitudeToDB('power', top_db=80)(torch.tensor(power_frames))

    return db_frames, rfft_bin_freq

#split a time series into multiple frames given a window size and step size
def get_even_time_series_frames(time_series : np.array, window_size : int, step_size : int):
    #calculate number of frames (has to be an even number?)
    frame_num = (len(time_series) - window_size)/step_size + 1
    series = time_series #copy series
    if frame_num % 2 != 0:  #ensure that number of frames is a even number
        frame_num = math.ceil(frame_num / 2.0) * 2 #round number of frames to even number
        padding_len = (frame_num - 1) * step_size + window_size - len(time_series)  #calculate padding length 
        series = np.pad(series, (0, padding_len), 'constant') #pad end of series with 0s
    frame_num = int(frame_num)
    #create and return frames from rime series 
    return np.array([ series[step*step_size:step*step_size+window_size] for step in range(frame_num)])

def apply_hanning_window(frames : np.array):
    frames *= np.hanning(frames.shape[frames.ndim - 1])

def calculate_power_spectrum(time_series_frames, n_fft = None):
    mag_frames = np.abs(np.fft.rfft(time_series_frames, n_fft)) # for each frame compute the n point fourier transform 
    pow_frames = ((1.0 / n_fft) * ((mag_frames) ** 2))
    return pow_frames

def get_frequency_bins(bin_num, sample_rate):
    return np.fft.rfftfreq(bin_num, d = 1.0/sample_rate)

def trim_to_frequency_range(frames, fft_bin_freq, max_freq, min_freq = 0):
    #get closest frequency in bin
    max_index = np.argmin(np.abs(fft_bin_freq - max_freq))
    min_index = np.argmin(np.abs(fft_bin_freq - min_freq))

    return frames[:,min_index:max_index], fft_bin_freq[min_index:max_index]

# Time Series to Spectogram ##########################################################################################
######################################################################################################################

######################################################################################################################
# Peak Detection #####################################################################################################

nb.jit('float64[:](float64[:], int, int, float32, float32, float32)', nopython=True, parallel=True)
def continuous_auxiliary_z_peak_detection(time_series : np.array, mean_lag_window : np.array, std_lag_window : np.array, mean_influence : float, std_influence : float, threshold : float):
    signal_peaks = np.zeros((time_series.shape))

    lag_window_mean = np.mean(mean_lag_window)
    lag_window_std = np.std(std_lag_window)

    for i in range(0, len(time_series)):
        mean_lag_window = np.roll(mean_lag_window, -1)
        std_lag_window = np.roll(std_lag_window, -1)

        if abs(time_series[i] - lag_window_mean) > threshold * lag_window_std:
            if time_series[i] > lag_window_mean: #check for positive or negative peak
                signal_peaks[i] = 1
            else:
                signal_peaks[i] = -1

            mean_lag_window[-1] = mean_influence * time_series[i] + (1 - mean_influence) * mean_lag_window[-2]            
            std_lag_window[-1] = std_influence * time_series[i] + (1 - std_influence) * std_lag_window[-2]            
        else:
            mean_lag_window[-1] = time_series[i]            
            std_lag_window[-1] = time_series[i]

        lag_window_mean = np.mean(mean_lag_window)            

    return signal_peaks, mean_lag_window, std_lag_window

nb.jit('float64[:,:](float64[:,:], float64[:,:], float64[:,:], float32, float32, float32)', nopython=True, parallel=True)
def continuous_auxiliary_z_score_peak_detection_2D(spectrogram : np.array, mean_lag_matrix : np.array, std_lag_matrix : np.array, mean_influence : float, std_influence : float, threshold : float):
    signal_peaks_matrix = np.zeros(spectrogram.shape)
    
    for freq_index in nb.prange(spectrogram.shape[0]):
        signal_peaks_matrix[freq_index], mean_lag_matrix[freq_index], std_lag_matrix[freq_index] = continuous_auxiliary_z_peak_detection(spectrogram[freq_index,:], mean_lag_matrix[freq_index,:], std_lag_matrix[freq_index,:], mean_influence, std_influence, threshold)           

    return signal_peaks_matrix

nb.jit('float64[:](float64[:], int, int, float32, float32, float32)', nopython=True, parallel=True)
def auxiliary_z_peak_detection(time_series : np.array, mean_lag : int, std_lag : int, mean_influence : float, std_influence : float, threshold : float):
    signal_peaks = np.zeros((time_series.shape))
    max_lag_length = mean_lag if mean_lag > std_lag else std_lag #fing the larger lag
    #get the windows for mean and std 
    mean_filtered_time_series_window = np.array(time_series[max_lag_length - mean_lag : max_lag_length])
    std_filtered_time_series_window = np.array(time_series[max_lag_length - std_lag : max_lag_length])

    lag_window_mean = np.mean(mean_filtered_time_series_window)
    lag_window_std = np.std(std_filtered_time_series_window)

    for i in range(max_lag_length, len(time_series)):
        mean_filtered_time_series_window = np.roll(mean_filtered_time_series_window, -1)
        std_filtered_time_series_window = np.roll(std_filtered_time_series_window, -1)

        if abs(time_series[i] - lag_window_mean) > threshold * lag_window_std:
            if time_series[i] > lag_window_mean: #check for positive or negative peak
                signal_peaks[i] = 1
            else:
                signal_peaks[i] = -1
            mean_filtered_time_series_window[-1] = mean_influence * time_series[i] + (1 - mean_influence) * mean_filtered_time_series_window[-2]            
            std_filtered_time_series_window[-1] = std_influence * time_series[i] + (1 - std_influence) * std_filtered_time_series_window[-2]            
        else:
            mean_filtered_time_series_window[-1] = time_series[i]            
            std_filtered_time_series_window[-1] = time_series[i]

        lag_window_mean = np.mean(mean_filtered_time_series_window)            

    return signal_peaks

nb.jit('float64[:,:](float64[:,:], int, int, float32, float32, float32)', nopython=True, parallel=True)
def auxiliary_z_score_peak_detection_2D(spectrogram : np.array, mean_lag : int, std_lag : int, mean_influence : float, std_influence : float, threshold : float):
    signal_peaks_matrix = np.zeros(spectrogram.shape)
    
    for freq_index in nb.prange(spectrogram.shape[0]):
        signal_peaks_matrix[freq_index] = auxiliary_z_peak_detection(spectrogram[freq_index,:], mean_lag, std_lag, mean_influence, std_influence, threshold)           

    return signal_peaks_matrix

# Peak Detection #####################################################################################################
######################################################################################################################

######################################################################################################################
# Audio Events #######################################################################################################

class audio_event: #simple class for storing the indices of an audio event
    def __init__(self, start_sec, end_sec, max_freq, min_freq): #initialize with start index
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.max_freq = max_freq
        self.min_freq = min_freq
    
    def get_values_as_array(self):
        return [self.start_sec, self.end_sec, self.max_freq, self.min_freq]

def get_cluster_scope(cluster_coordinates):
    return np.min(cluster_coordinates[:,1]), np.max(cluster_coordinates[:,1]), np.min(cluster_coordinates[:,0]), np.max(cluster_coordinates[:,0])

def get_audio_event_scopes_from_hdbscan(peak_coordinates, min_cluster_size):
    clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size).fit(peak_coordinates)
    clusters_coordinates = [[] for _ in range(len(set(clusterer.labels_)))]

    for index, cluster_index in enumerate(clusterer.labels_):
        if cluster_index != -1:
            clusters_coordinates[cluster_index].append(peak_coordinates[index])

    cluster_scopes = np.array([np.asarray(get_cluster_scope(np.array(coordinates))) for coordinates in clusters_coordinates if coordinates])

    return  cluster_scopes, clusterer

def do_events_overlap(top_left_corner_1: tuple, bottom_right_corner_1: tuple, top_left_corner_2: tuple, bottom_right_corner_2: tuple):
    if (top_left_corner_1[0] >= bottom_right_corner_2[0] or top_left_corner_2[0] >= bottom_right_corner_1[0]):
        return False
    
    if (top_left_corner_1[1] <= bottom_right_corner_2[1] or top_left_corner_2[1] <= bottom_right_corner_1[1]):
        return False

    return True

def concatenate_events(event_arr, event_processing_parameters):
    '''
    event_arr has to be sorted sequentially by the start time of events
    '''
    event_distance_max = event_processing_parameters.event_distance_max
    event_freq_differnce_max = event_processing_parameters.event_freq_differnce_max
    new_event_arr = [event_arr[0]]
    for event in event_arr[1:]:
        if do_events_overlap((event[0] - event_distance_max, event[3] + event_freq_differnce_max), (event[1], event[2] - event_freq_differnce_max),
         (new_event_arr[-1][0], new_event_arr[-1][3]), (new_event_arr[-1][1], new_event_arr[-1][2])):
            new_event_arr[-1][1] = event[1]
            new_event_arr[-1][2] = min((event[2], new_event_arr[-1][2]))
            new_event_arr[-1][3] = max((event[3], new_event_arr[-1][3]))

        #if ((event_arr[i][0] - new_event_arr[-1][1]) < event_distance_max) :
        #    new_event_arr[-1][1] = event_arr[i][1]
        else:
            new_event_arr.append(event)
    return new_event_arr

def index_scopes_to_unit_scopes(event_scopes, sample_rate, step_size, rfft_freq_bins):
    return [[scope[0] * step_size / sample_rate, scope[1] * step_size / sample_rate, rfft_freq_bins[scope[2]], rfft_freq_bins[scope[3]]] for scope in event_scopes]

def unit_time_to_index(event_time, sample_rate):
    return int(event_time * sample_rate)

def audio_events_to_csv(events, sample_id):
    with open(os.path.join(audio_events_dir, str(sample_id) + '.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['start_sec', 'end_sec', 'min_freq', 'max_freq'])
        for event in events:
            writer.writerow(event)

# Audio Events #######################################################################################################
######################################################################################################################

# region plotting 
######################################################################################################################
# plotting ###########################################################################################################

def save_spectrogram_plot(spectrogram_matrix, sample_rate, step_size, sample_id, title = 'Spectrogram', x_label = None, y_labels = None, y_tick_num = 6, audio_events = []):
    f = plt.figure(figsize=(10,5), dpi= 80)
    ax = f.add_subplot()
    ax.set_title(title) #set title 
    spectrogram = ax.matshow(spectrogram_matrix, aspect="auto", cmap=plt.get_cmap('magma')) #draw matrix with colormap 'magma'

    if audio_events:
        colors = cm.rainbow(np.linspace(0, 1, len(audio_events)))
        for index, event in enumerate(audio_events):
            ax.add_patch(patches.Rectangle((event[0], event[2]), event[1] - event[0], event[3] - event[2], linewidth=1, edgecolor=colors[index], facecolor='none'))


    ax.xaxis.set_ticks_position('bottom') #set x ticks to bottom of graph 
    ax.set_xlabel('Time (sec)')
    locator_num = 16 if spectrogram_matrix.shape[1] * step_size // sample_rate >= 16 else spectrogram_matrix.shape[1] * step_size // sample_rate
    ax.set_xlim(left = 0, right = spectrogram_matrix.shape[1])
    ax.xaxis.set_major_locator(ticker.LinearLocator(locator_num))
    formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%-S', time.gmtime(ms * step_size // sample_rate)))
    ax.xaxis.set_major_formatter(formatter)
    
    ax.invert_yaxis() #have y axis start from the bottom
    ax.set_ylabel('Hz')
    y_tick_steps = int(len(y_labels) / y_tick_num)
    ax.set_yticks(np.arange(0, len(y_labels), y_tick_steps))
    ax.set_yticklabels(y_labels[0::y_tick_steps])

    plt.tight_layout()
    plt.colorbar(spectrogram, format='%+2.0f dB')
    plt.savefig(os.path.join(plots_path, str(sample_id), title + '.png'))


def save_cluster_plot(peak_coordinates, sample_rate, step_size, clusterer, sample_id, x_dim, y_dim, y_labels, y_tick_num = 6,):
    def adjust_alpha(color, alpha):
        color[-1] = alpha
        return color

    cluster_colors = plt.cm.Spectral(np.linspace(0, 1, len(set(clusterer.labels_))-1))

    colors = [cluster_colors[x] if x >= 0 else [0.5, 0.5, 0.5, 0.0] for x in clusterer.labels_]

    colors_with_probs = [ adjust_alpha(x, p) for x, p in zip(colors, clusterer.probabilities_)]

    f = plt.figure(figsize=(10,5), dpi = 80)
    ax = f.add_subplot()
    ax.set_title(f"HDBScan number of clusters: {(len(set(clusterer.labels_)) - 1)}") #set title 
    ax.set_facecolor('dimgray')
    
    ax.xaxis.set_ticks_position('bottom') #set x ticks to bottom of graph 
    ax.set_xlabel('Time (sec)')
    locator_num = 16 if x_dim * step_size // sample_rate >= 16 else x_dim * step_size // sample_rate
    ax.set_xlim(left = 0, right = x_dim)
    ax.xaxis.set_major_locator(ticker.LinearLocator(locator_num))
    formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%-S', time.gmtime(ms * step_size // sample_rate)))
    ax.xaxis.set_major_formatter(formatter)

    ax.set_ylabel('Hz')
    ax.set_ylim(0, y_dim)
    y_tick_steps = int(len(y_labels) / y_tick_num)
    ax.set_yticks(np.arange(0, len(y_labels), y_tick_steps))
    ax.set_yticklabels(y_labels[0::y_tick_steps])
    scatter_plot = ax.scatter(peak_coordinates[:,1], peak_coordinates[:,0], s=2, linewidth=0, c=colors_with_probs)

    plt.tight_layout()
    plt.colorbar(scatter_plot)
    plt.savefig(os.path.join(plots_path, str(sample_id), "Clusters.png"))

def show_amplitude_wave_plot(time_series, sample_rate):
    '''
    plots the amplitude vs time of the sample 
    '''
    fig, ax = plt.subplots(figsize=(10,5), dpi= 80)


    ax.plot(time_series)

    locator_num = 16 if len(time_series) // sample_rate >= 16 else len(time_series) // sample_rate
    ax.set_xlim(left = 0, right = len(time_series))
    ax.xaxis.set_major_locator(ticker.LinearLocator(locator_num))
    formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%-S', time.gmtime(ms // sample_rate)))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Amplitude')
    plt.show()

# plotting ###########################################################################################################
######################################################################################################################
# endregion

# region audio processing
######################################################################################################################
# audio processing ###################################################################################################

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/np.amax(np.abs(tensor_minusmean))

def process_spectrogram(db_frames, rfft_bin_freq, audio_processing_parameters):
    #test #######
    relevant_freq_range = audio_processing_parameters.relevant_freq_range #always use uneven number
    time_dim = db_frames.shape[0]
    freq_dim = db_frames.shape[1]
    freq_padding = int(relevant_freq_range/2)
    accumulated_frames = np.zeros((time_dim, freq_dim - freq_padding * 2))

    for time_index in range(time_dim):
        for freq_index in range(freq_padding, freq_dim - freq_padding):
            accumulated_frames[time_index][freq_index - freq_padding] = np.median(db_frames[time_index][freq_index - freq_padding : freq_index + freq_padding])
    #test #######

    return accumulated_frames, rfft_bin_freq[freq_padding:freq_dim - freq_padding]

def detect_peaks_in_spectrogram(spectrogram, event_detection_parameters):
    spectrogram_peaks = auxiliary_z_score_peak_detection_2D(spectrogram, event_detection_parameters.mean_lag_window_size, event_detection_parameters.std_lag_window_size, event_detection_parameters.mean_influence, event_detection_parameters.std_influence, event_detection_parameters.threshold)
    #remove negative peaks
    return np.clip(spectrogram_peaks, 0, 1)

def cluster_audio_events_from_coordinates(peak_coordinates, clustering_parameters):
    #group single peaks into audio events
    audio_event_index_scopes, clusterer = get_audio_event_scopes_from_hdbscan(peak_coordinates, clustering_parameters.min_cluster_size)

    #sort audio events by their occurrence 
    audio_event_index_scopes = sorted(audio_event_index_scopes, key = lambda x: x[1])
    #process audio events
    return audio_event_index_scopes, peak_coordinates, clusterer


def cluster_audio_events(spectrogram_peaks, clustering_parameters):
    #get peak coordinates
    peak_coordinates = np.array(np.where(spectrogram_peaks == 1)).T
    #group single peaks into audio events
    audio_event_index_scopes, clusterer = get_audio_event_scopes_from_hdbscan(peak_coordinates, clustering_parameters.min_cluster_size)

    #sort audio events by their occurrence 
    audio_event_index_scopes = sorted(audio_event_index_scopes, key = lambda x: x[1])
    #process audio events
    return audio_event_index_scopes, peak_coordinates, clusterer

# audio processing ###################################################################################################
######################################################################################################################
# endregion


######################################################################################################################
# main functions #####################################################################################################

def custom_detect_audio_events_in_sample( process_function, 
    sample_id, 
    audio_conversion_parameters, 
    audio_processing_parameters,
    event_detection_parameters, 
    clustering_parameters, 
    event_processing_parameters, 
    additional_parameters,
     ):

    if additional_parameters.generate_process_plots:
        empty_or_create_dir(os.path.join(plots_path, str(sample_id)))

    audio_events = process_function(sample_id, audio_conversion_parameters, audio_processing_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters)

    audio_events_to_csv(audio_events, sample_id)

def detect_audio_events_in_sample( process_function, 
    sample_id, 
    audio_conversion_parameters, 
    event_detection_parameters, 
    clustering_parameters, 
    event_processing_parameters, 
    additional_parameters,
     ):

    if additional_parameters.generate_process_plots:
        empty_or_create_dir(os.path.join(plots_path, str(sample_id)))

    audio_events = process_function(sample_id, audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters)

    audio_events_to_csv(audio_events, sample_id)

def detect_audio_events_with_custom_chunked(
    sample_id, 
    audio_conversion_parameters, 
    audio_processing_parameters,
    event_detection_parameters, 
    clustering_parameters, 
    event_processing_parameters, 
    additional_parameters ):
    

    #load sample as time series array
    sample, _ = load_audio_sample(sample_id, audio_conversion_parameters.sample_rate)

    chunk_size = 44100 * 10

    for chunk_index in range(0, len(sample), chunk_size):

        audio_chunk = sample[chunk_index: chunk_index + chunk_size]

        db_spectrogram, rfft_bin_freq = create_spectrogram(audio_chunk, audio_conversion_parameters.window_size, audio_conversion_parameters.step_size, audio_conversion_parameters.sample_rate)
        #trim the spectrogram to a specified frequency range
        db_spectrogram, rfft_bin_freq = trim_to_frequency_range(db_spectrogram, rfft_bin_freq, audio_conversion_parameters.max_frequency, audio_conversion_parameters.min_frequency)

        if additional_parameters.generate_process_plots:    #save the decibel spectrogram
            save_spectrogram_plot(np.array(db_spectrogram).T, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram (Decibel)", y_labels = rfft_bin_freq)

        db_spectrogram, rfft_bin_freq = audio_processing_parameters.fn_process_spectrogram(db_spectrogram, rfft_bin_freq, audio_processing_parameters)    

        transposed_db_spectrogram = np.array(db_spectrogram).T #transpose spectrogram frames dimension from (time_step, frequency) to (frequency, time_step)

        if additional_parameters.generate_process_plots:    #save the decibel spectrogram
            save_spectrogram_plot(transposed_db_spectrogram, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Post processing Spectogram (Decibel)", y_labels = rfft_bin_freq)

        #detect peaks in spectrogram
        spectrogram_peaks = event_detection_parameters.fn_detect_peaks_in_spectrogram(transposed_db_spectrogram, event_detection_parameters)

        if additional_parameters.generate_process_plots: #save_spectrogram_peaks()
            save_spectrogram_plot(spectrogram_peaks, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram Peaks", y_labels = rfft_bin_freq)

        #accumulate individual audio peaks to sound events
        audio_event_index_scopes, peak_coordinates, clusterer = clustering_parameters.fn_cluster_audio_events(spectrogram_peaks, clustering_parameters)

        if additional_parameters.generate_process_plots: #save cluster plot
            save_cluster_plot(peak_coordinates, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, clusterer, sample_id, spectrogram_peaks.shape[1], spectrogram_peaks.shape[0], rfft_bin_freq,)

    audio_event_index_scopes = concatenate_events(audio_event_index_scopes, event_processing_parameters)

    if additional_parameters.generate_process_plots: #save the decibel spectrogram with audio event boxes
        save_spectrogram_plot(transposed_db_spectrogram, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram with Events", y_labels = rfft_bin_freq, audio_events=audio_event_index_scopes)
    #convert audio event index scopes to audio events with time and frequency scopes
    audio_events = index_scopes_to_unit_scopes(audio_event_index_scopes, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, rfft_bin_freq)
    #save events
    return audio_events

def detect_audio_events_with_custom(
    sample_id, 
    audio_conversion_parameters, 
    audio_processing_parameters,
    event_detection_parameters, 
    clustering_parameters, 
    event_processing_parameters, 
    additional_parameters ):
    
    #load sample as time series array
    sample, _ = load_audio_sample(sample_id, audio_conversion_parameters.sample_rate)

    sample = normalize(sample)

    db_frames, rfft_bin_freq = create_spectrogram(sample, audio_conversion_parameters.window_size, audio_conversion_parameters.step_size, audio_conversion_parameters.sample_rate)
    #trim the spectrogram to a specified frequency range
    db_frames, rfft_bin_freq = trim_to_frequency_range(db_frames, rfft_bin_freq, audio_conversion_parameters.max_frequency, audio_conversion_parameters.min_frequency)

    if additional_parameters.generate_process_plots:    #save the decibel spectrogram
        save_spectrogram_plot(np.array(db_frames).T, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram (Decibel)", y_labels = rfft_bin_freq)

    db_frames, rfft_bin_freq = audio_processing_parameters.fn_process_spectrogram(db_frames, rfft_bin_freq, audio_processing_parameters)    

    transposed_db_spectrogram = np.array(db_frames).T #transpose spectrogram frames dimension from (time_step, frequency) to (frequency, time_step)

    if additional_parameters.generate_process_plots:    #save the decibel spectrogram
        save_spectrogram_plot(transposed_db_spectrogram, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Post processing Spectogram (Decibel)", y_labels = rfft_bin_freq)

    #detect peaks in spectrogram
    spectrogram_peaks = event_detection_parameters.fn_detect_peaks_in_spectrogram(transposed_db_spectrogram, event_detection_parameters)

    if additional_parameters.generate_process_plots: #save_spectrogram_peaks()
        save_spectrogram_plot(spectrogram_peaks, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram Peaks", y_labels = rfft_bin_freq)

    #accumulate individual audio peaks to sound events
    audio_event_index_scopes, peak_coordinates, clusterer = clustering_parameters.fn_cluster_audio_events(spectrogram_peaks, clustering_parameters)

    if additional_parameters.generate_process_plots: #save cluster plot
        save_cluster_plot(peak_coordinates, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, clusterer, sample_id, spectrogram_peaks.shape[1], spectrogram_peaks.shape[0], rfft_bin_freq,)

    audio_event_index_scopes = concatenate_events(audio_event_index_scopes, event_processing_parameters)

    if additional_parameters.generate_process_plots: #save the decibel spectrogram with audio event boxes
        save_spectrogram_plot(transposed_db_spectrogram, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram with Events", y_labels = rfft_bin_freq, audio_events=audio_event_index_scopes)
    #convert audio event index scopes to audio events with time and frequency scopes
    audio_events = index_scopes_to_unit_scopes(audio_event_index_scopes, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, rfft_bin_freq)
    #save events
    return audio_events


def detect_audio_events_with_zscore_chunked(
    sample_id, 
    audio_conversion_parameters, 
    event_detection_parameters, 
    clustering_parameters, 
    event_processing_parameters, 
    additional_parameters ):

    #load sample as time series array
    sample, _ = load_audio_sample(sample_id, audio_conversion_parameters.sample_rate)

    sample = normalize(sample)

    chunk_size = audio_conversion_parameters.sample_rate * audio_conversion_parameters.chunk_len_sec #seconds

    mean_lag_length = event_detection_parameters.mean_lag_window_size
    std_lag_length = event_detection_parameters.std_lag_window_size
    max_lag_length = mean_lag_length if mean_lag_length > std_lag_length else std_lag_length #fing the larger lag

    if max_lag_length >= chunk_size:
        raise Exception(f"Chunk size '{chunk_size}' is not large enough for the maximum lag length {max_lag_length}")

    print(f"Total sample len: {len(sample)} vs chunk size: {chunk_size} ")

    #get the windows for mean and std 
    mean_window_matrix = None
    std_window_matrix = None

    all_audio_event_index_scopes = []

    is_first_chunk = True

    full_spectrogram = None

    total_frame_num = 0

    for chunk_index in range(0, len(sample), chunk_size):

        audio_chunk = sample[chunk_index: chunk_index + chunk_size]

        db_frames, rfft_bin_freq = create_spectrogram(audio_chunk, audio_conversion_parameters.window_size, audio_conversion_parameters.step_size, audio_conversion_parameters.sample_rate)
        #trim the spectrogram to a specified frequency range
        db_frames, rfft_bin_freq = trim_to_frequency_range(db_frames, rfft_bin_freq, audio_conversion_parameters.max_frequency, audio_conversion_parameters.min_frequency)

        frame_len = len(db_frames)
        #db_frames = process_spectrogram(db_frames)    

        transposed_db_spectrogram = np.array(db_frames).T #transpose spectrogram frames dimension from (time_step, frequency) to (frequency, time_step)

        if additional_parameters.generate_process_plots:    #save the decibel spectrogram
            save_spectrogram_plot(transposed_db_spectrogram, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram (Decibel)", y_labels = rfft_bin_freq)

        if is_first_chunk:
            is_first_chunk = False
            mean_window_matrix = np.copy(transposed_db_spectrogram[:,max_lag_length - mean_lag_length : max_lag_length])
            std_window_matrix = np.copy(transposed_db_spectrogram[:,max_lag_length - std_lag_length : max_lag_length])
            full_spectrogram = transposed_db_spectrogram
            transposed_db_spectrogram = transposed_db_spectrogram[:, max_lag_length:]
        
        elif additional_parameters.generate_process_plots:
            full_spectrogram = np.hstack((full_spectrogram, transposed_db_spectrogram))
            max_lag_length = 0

        else:
            max_lag_length = 0
        
        #detect peaks in spectrogram
        spectrogram_peaks = continuous_auxiliary_z_score_peak_detection_2D(transposed_db_spectrogram, mean_window_matrix, std_window_matrix, event_detection_parameters.mean_influence, event_detection_parameters.std_influence, event_detection_parameters.threshold)
        spectrogram_peaks = np.clip(spectrogram_peaks, 0, 1)

        if additional_parameters.generate_process_plots: #save_spectrogram_peaks()
            save_spectrogram_plot(spectrogram_peaks, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram Peaks", y_labels = rfft_bin_freq)

        #get peak coordinates
        peak_coordinates = np.array(np.where(spectrogram_peaks == 1)).T


        peak_coordinates[:, 1] = peak_coordinates[:, 1] + total_frame_num + max_lag_length

        #accumulate individual audio peaks to sound events
        audio_event_index_scopes, peak_coordinates, clusterer = cluster_audio_events_from_coordinates(peak_coordinates, clustering_parameters)

        all_audio_event_index_scopes += audio_event_index_scopes


        total_frame_num += frame_len
        # if additional_parameters.generate_process_plots: #save cluster plot
        #     readjusted_peak_coordinates = np.copy(peak_coordinates)
        #     readjusted_peak_coordinates[:, 1] = readjusted_peak_coordinates[:, 1] - chunk_index - max_lag_length
        #     save_cluster_plot(peak_coordinates, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, clusterer, sample_id, spectrogram_peaks.shape[1], spectrogram_peaks.shape[0], rfft_bin_freq,)

    if additional_parameters.generate_process_plots: #save_spectrogram_peaks()
        all_spectrogram_peaks = np.zeros(full_spectrogram.shape)
            

        save_spectrogram_plot(all_spectrogram_peaks, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram Peaks", y_labels = rfft_bin_freq)

    all_audio_event_index_scopes = concatenate_events(all_audio_event_index_scopes, event_processing_parameters)

    
    if additional_parameters.generate_process_plots: #save the decibel spectrogram with audio event boxes
        save_spectrogram_plot(full_spectrogram, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram with Events", y_labels = rfft_bin_freq, audio_events=all_audio_event_index_scopes)
    
    #convert audio event index scopes to audio events with time and frequency scopes
    audio_events = index_scopes_to_unit_scopes(all_audio_event_index_scopes, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, rfft_bin_freq)
    #save events
    return audio_events

def detect_audio_events_with_zscore(
    sample_id, 
    audio_conversion_parameters, 
    event_detection_parameters, 
    clustering_parameters, 
    event_processing_parameters, 
    additional_parameters ):

    #load sample as time series array
    sample, _ = load_audio_sample(sample_id, audio_conversion_parameters.sample_rate)

    db_frames, rfft_bin_freq = create_spectrogram(sample, audio_conversion_parameters.window_size, audio_conversion_parameters.step_size, audio_conversion_parameters.sample_rate)
    #trim the spectrogram to a specified frequency range
    db_frames, rfft_bin_freq = trim_to_frequency_range(db_frames, rfft_bin_freq, audio_conversion_parameters.max_frequency, audio_conversion_parameters.min_frequency)

    #db_frames = process_spectrogram(db_frames)    

    transposed_db_spectrogram = np.array(db_frames).T #transpose spectrogram frames dimension from (time_step, frequency) to (frequency, time_step)

    if additional_parameters.generate_process_plots:    #save the decibel spectrogram
        save_spectrogram_plot(transposed_db_spectrogram, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram (Decibel)", y_labels = rfft_bin_freq)

    #detect peaks in spectrogram
    spectrogram_peaks = detect_peaks_in_spectrogram(transposed_db_spectrogram, event_detection_parameters)

    if additional_parameters.generate_process_plots: #save_spectrogram_peaks()
        save_spectrogram_plot(spectrogram_peaks, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram Peaks", y_labels = rfft_bin_freq)

    #accumulate individual audio peaks to sound events
    audio_event_index_scopes, peak_coordinates, clusterer = cluster_audio_events(spectrogram_peaks, clustering_parameters)

    if additional_parameters.generate_process_plots: #save cluster plot
        save_cluster_plot(peak_coordinates, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, clusterer, sample_id, spectrogram_peaks.shape[1], spectrogram_peaks.shape[0], rfft_bin_freq,)

    audio_event_index_scopes = concatenate_events(audio_event_index_scopes, event_processing_parameters)
    
    if additional_parameters.generate_process_plots: #save the decibel spectrogram with audio event boxes
        save_spectrogram_plot(transposed_db_spectrogram, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram with Events", y_labels = rfft_bin_freq, audio_events=audio_event_index_scopes)
    
    #convert audio event index scopes to audio events with time and frequency scopes
    audio_events = index_scopes_to_unit_scopes(audio_event_index_scopes, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, rfft_bin_freq)
    #save events
    return audio_events

# main functions #####################################################################################################
######################################################################################################################
def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Download mp3 samples from species described int the data_dictionary directory')
    parser.add_argument('--all_samples', required=False, action='store_true',
                            help =  'If this argument is set, all samples from download_species_sample_info.json will be processed')
    parser.add_argument('--sample_ids', type=int, required=False, default=[], nargs='+', 
                            help =  'Specify by id which samples should be processed.')
    parser.add_argument('--sample_ids_from_file', type=str, required=False, 
                            help =  'Give a path to a txt file which specifies sample ids which should be processed.')
    parser.add_argument('--generate_process_plots', required=False, action='store_true',
                            help =  'If this argument is set, processing steps will be visualized and saved to audio_event_plots (for testing)')
    parser.add_argument('--multi_processing', type=int, required=False, nargs='?', const=os.cpu_count(),
                            help =  'Specify number of cores that should be used')
    args =  parser.parse_args()

    if not os.path.exists(audio_events_dir):
        os.mkdir(audio_events_dir)

    if args.generate_process_plots:
        empty_or_create_dir(plots_path)

    if args.all_samples:
        sample_id_arr = get_all_samples_ids(raw_sample_dir)
    elif args.sample_ids:
        sample_id_arr = args.sample_ids
    elif args.sample_ids_from_file:
        print("Feature is note yet implemented")
    else: 
        raise Exception("Please provide samples through '--sample_ids', '--sample_ids_from_file', or '--all_samples'")

    return args, sample_id_arr

def custom_main(process_function, audio_conversion_parameters, audio_processing_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters,):
    args, sample_id_arr = parse_input_arguments()

    additional_parameters = AdditionalParameters(args.generate_process_plots)
    
    if args.multi_processing:
        print(f"Using {args.multi_processing} cores for multi processing!")
        with multiprocessing.Pool(processes=args.multi_processing) as pool:
            pool.starmap(custom_detect_audio_events_in_sample, [(process_function, sample_id, audio_conversion_parameters, audio_processing_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters) for sample_id in sample_id_arr])
    else:
        for sample_id in sample_id_arr:
            custom_detect_audio_events_in_sample(process_function, sample_id, audio_conversion_parameters, audio_processing_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters)


def main(process_function, audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, ):
    args, sample_id_arr = parse_input_arguments()

    additional_parameters = AdditionalParameters(args.generate_process_plots)
    
    if args.multi_processing:
        print(f"Using {args.multi_processing} cores for multi processing!")
        with multiprocessing.Pool(processes=args.multi_processing) as pool:
            pool.starmap(detect_audio_events_in_sample, [(process_function, sample_id, audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters) for sample_id in sample_id_arr])
    else:
        for sample_id in sample_id_arr:
            detect_audio_events_in_sample(process_function, sample_id, audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters)


if __name__ == '__main__':
    #Variables for audio detection
    sample_rate = 44100 # 44.1 kHz
    ##Spectogram variables
    window_size = 2048
    step_size = 512

    ##Frequency trims variables
    max_frequency = 10000
    min_frequency = 1000

    ##z score peak detection variables
    mean_lag_window_size = 128
    std_lag_window_size = 128
    mean_influence = .001
    std_influence = .001
    threshold = 2.5

    #HDBScan vars 
    min_cluster_size = 25

    #Audio Event processing 
    #Max distance between individual events to be counted together  
    event_distance_max = 0#int(0.1 * sample_rate) # 100ms * sample rate for the max audio event distance
    event_freq_differnce_max = 0
    event_length_min = int(0.125 * sample_rate) # 125ms * sample rate will give us the max distance in "index" units

    #Events to audio variables
    #set half a sec as buffer lengths
    start_buffer_len = int(.5 * sample_rate)
    end_buffer_len = int(.5 * sample_rate)

    audio_conversion_parameters = AudioConversionParameters(sample_rate, window_size, step_size, max_frequency, min_frequency)
    event_detection_parameters = EventDetectionParameters(mean_lag_window_size, std_lag_window_size, mean_influence, std_influence, threshold)
    clustering_parameters = ClusteringParameters(min_cluster_size)
    event_processing_parameters = EventProcessingParameters(event_distance_max, event_freq_differnce_max, event_length_min, start_buffer_len, end_buffer_len)

    main(detect_audio_events_with_zscore, audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters)