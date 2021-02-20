import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
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


#Variables for audio detection

main_sample_rate = 44100 # 44.1 kHz
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
event_distance_max = int(0.1 * main_sample_rate) # 100ms * sample rate for the max audio event distance
event_length_min = int(0.125 * main_sample_rate) # 125ms * sample rate will give us the max distance in "index" units

#Events to audio variables
#set half a sec as buffer lengths
start_buffer_len = int(.5 * main_sample_rate)
end_buffer_len = int(.5 * main_sample_rate)

AudioConversionParameters = namedtuple('AudioConversionParameters', ['main_sample_rate', 'window_size', 'step_size', 'max_frequency', 'min_frequency'])
EventDetectionParameters = namedtuple('EventDetectionParameters', ['mean_lag_window_size', 'std_lag_window_size', 'mean_influence', 'std_influence', 'threshold'])
ClusteringParameters = namedtuple('ClusteringParameters', ['min_cluster_size'])
EventProcessingParameters = namedtuple('EventProcessingParameters', ['event_distance_max', 'event_length_min', 'start_buffer_len', 'end_buffer_len'])
AdditionalParameters = namedtuple('AdditionalParameters', 'generate_process_plots')

audio_conversion_parameters = AudioConversionParameters(main_sample_rate, window_size, step_size, max_frequency, min_frequency)
event_detection_parameters = EventDetectionParameters(mean_lag_window_size, std_lag_window_size, mean_influence, std_influence, threshold)
clustering_parameters = ClusteringParameters(min_cluster_size)
event_processing_parameters = EventProcessingParameters(event_distance_max, event_length_min, start_buffer_len, end_buffer_len)

######################################################################################################################
######################################################################################################################
species_info_df = None # dict with species info

file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

raw_sample_dir = os.path.join(file_dir, 'dataset', 'raw') #path pointing to samples
#processed_sample_dir = os.path.join(file_dir, 'dataset', 'processed') #path pointing to samples
audio_events_dir = os.path.join(file_dir, 'dataset', 'audio_events') #path pointing to samples
samples_metadata_file_path = os.path.join(file_dir, 'dataset', 'data_dictionary', 'samples_metadata.json')
download_species_sample_path = os.path.join(file_dir, 'dataset', 'data_dictionary', 'download_species_sample_info.json')
plots_path = os.path.join(file_dir, 'audio_event_plots')

######################################################################################################################
######################################################################################################################

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
        os.mkdir(dir_path)


def get_all_samples_ids(dir_path):
    return [os.path.splitext(f)[0] for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    #sample_ids = [f for f in listdir(raw_sample_dir) if isfile(join(raw_sample_dir, f))]
    #return sample_ids

###############################################
#TO BE REMOVED
def get_all_samples_dir(dir_path):
    return [os.path.splitext(f)[0] for f in listdir(dir_path) if isfile(os.path.join(dir_path, f))]

def get_sample_metadata():
    '''
    loads the file that holds information on the downloaded samples and the relevant species
    '''
    with open(samples_metadata_file_path) as f:
        samples_metadata = json.load(f)
    return samples_metadata

def get_species_info():
    '''
    loads the file that holds information on all species
    '''
    return pd.read_csv(species_info_path).set_index('species_key')

def display_sample_metadata(sample_id):
    '''
    displays the meta data + some additional info on the species in the sample. Takes the sample id as a parameter 
    '''
    samples_metadata = get_sample_metadata()

    sample_info = samples_metadata.get(str(sample_id), None)
    if sample_info is None:
        raise Exception("sample with id '{}' was not found in sample_metadata".format(sample_id))
    print('-' * 90)
    print(f"Sample bgifID: {sample_info['gbifID']:>10}")
    print(f"url: {sample_info['recording_link']}")
    print(f"rec time (sec): {sample_info['recording_time_sec']}")
    print(f"rec date: {sample_info['date']}")
    print()
    print(f"decimal latitude: {sample_info['decimal_latitude']}")
    print(f"decimal longitude: {sample_info['decimal_longitude']}")
    main_species_key = sample_info['forefront_bird_key']
    print()
    print(f"main species key: {main_species_key}".format())
    print(f"main scientific name: {species_info_df.at[main_species_key, 'scientific_name']}")
    print(f"main common name: {species_info_df.at[main_species_key, 'common_name']}")
    print()
    if sample_info['behavior']:
        print("Noted behavior in this sample:")
        for index, behavior in enumerate(sample_info['behavior']):
            print(f"\t{str(index)+')'} {behavior}")
    else:
        print("Noted behavior in this sample: None")
    if sample_info['background_birds_keys']:
        print("Background bird's species keys and info:")
        print(f"\t{'':3} {'key':^10} | {'scientific name':30} | {'common_name':30}")
        print(f"\t{'-'*15:^15}┼{'-'*32:32}┼{'-'*30:30}")
        for index, key in enumerate(sample_info['background_birds_keys']):
            print(f"\t{str(index)+')':3} {key:10} | {species_info_df.at[key, 'scientific_name']:30} | {species_info_df.at[key, 'common_name']:30}")
    else:
        print("Background bird's species keys: None")
    print('-' * 90)

#TO BE REMOVED
###############################################

def load_audio_sample(sample_id, main_sample_rate):
    '''
    loads an audio sample by it's id (file name). Returns the audio and the the sample rate
    '''
    audio_file_path = os.path.join(raw_sample_dir, str(sample_id) + '.mp3')
    if not os.path.isfile(audio_file_path): #check if file exists
        raise Exception(f"Sample with id '{sample_id}' does not exist! Available files can be found in the {raw_sample_dir} directory")
    #sample audio at 44.1 khz and get the time series as a numpy array
    time_series, sample_rate = librosa.load(audio_file_path, sr = main_sample_rate)
    #trim empty start and end padding from time series
    time_series, _ = librosa.effects.trim(time_series)

    return time_series, sample_rate

###############################################
#TO BE REMOVED

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

def show_mel_spectogram(sample_time_series):
    '''
    plots the melspectogram of the sample
    '''
    S = librosa.feature.melspectrogram(y=sample_time_series, sr=main_sample_rate, n_mels=128, fmax=10000, center = False)
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=main_sample_rate,
                            fmax=10000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()


#TO BE REMOVED
###############################################


######################################################################################################################
# Time Series to Spectogram ##########################################################################################

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
        std_window_mean = np.std(std_filtered_time_series_window)

    return signal_peaks

def auxiliary_z_score_peak_detection_2D(frequency_time_series : np.array, mean_lag : int, std_lag : int, mean_influence : float, std_influence : float, threshold : float):
    return np.asarray([auxiliary_z_peak_detection(freq_series, mean_lag, std_lag, mean_influence, std_influence, threshold) for freq_series in frequency_time_series])

# Peak Detection #####################################################################################################
######################################################################################################################

######################################################################################################################
# Audio Events #######################################################################################################

class audio_event: #simple class for storing the indices of an audio event
    def __init__(self, start_sec, end_sec, max_freq, min_freq): #initialize with start index
        self.start_sec = start_index
        self.end_sec = end_sec
        self.max_freq = max_freq
        self.min_freq = min_freq
    
    def get_values_as_array(self):
        return [self.start_sec, self.end_sec, self.max_freq, self.min_freq]

def get_cluster_scope(cluster_coordinates):
    return np.min(cluster_coordinates[:,1]), np.max(cluster_coordinates[:,1]), np.min(cluster_coordinates[:,0]), np.max(cluster_coordinates[:,0])

def get_audio_event_scopes_from_hdbscan(peak_coordinates, save_process_step_plots = False, sample_id = None):

    #change function to not save plot, saving the plot should be done externally

    clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size).fit(peak_coordinates)
    clusters_coordinates = [[] for i in range(len(set(clusterer.labels_)) - 1)]

    if save_process_step_plots: 
        def adjust_alpha(color, alpha):
            color[-1] = alpha
            return color

        cluster_colors = plt.cm.Spectral(np.linspace(0, 1, len(set(clusterer.labels_))-1))

        colors = [cluster_colors[x] if x >= 0 else [0.5, 0.5, 0.5, 0.0] for x in clusterer.labels_]

        colors_with_probs = [ adjust_alpha(x, p) for x, p in zip(colors, clusterer.probabilities_)]

        f = plt.figure(figsize=(10,5), dpi = 80)
        ax = f.add_subplot()
        ax.set_title(f"HDBScan number of clusters: {(len(set(clusterer.labels_)) - 1)}") #set title 
        ax.set_facecolor('black')
        ax.scatter(peak_coordinates[:,1], peak_coordinates[:,0], s=2, linewidth=0, c=colors_with_probs)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, str(sample_id), "Clusters.png"))


    for index, cluster_index in enumerate(clusterer.labels_):
        if cluster_index != -1:
            clusters_coordinates[cluster_index].append(peak_coordinates[index])

    cluster_scopes = np.array([np.asarray(get_cluster_scope(np.array(coordinates))) for coordinates in clusters_coordinates])

    return  cluster_scopes

def do_events_overlap(top_left_corner_1: tuple, bottom_right_corner_1: tuple, top_left_corner_2: tuple, bottom_right_corner_2: tuple):
    if (top_left_corner_1[0] >= bottom_right_corner_2[0] or top_left_corner_2[0] >= bottom_right_corner_1[0]):
        return False
    
    if (top_left_corner_1[1] <= bottom_right_corner_2[1] or top_left_corner_2[1] <= bottom_right_corner_1[1]):
        return False

    return True

def concatenate_events(event_arr, event_distance_max = 0, event_freq_differnce_max = 0):
    '''
    event_arr has to be sorted sequentially by the start time of events
    '''
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

def audio_events_to_csv(events, sample_id):
    with open(os.path.join(audio_events_dir, str(sample_id) + '.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['start_sec', 'end_sec', 'min_freq', 'max_freq'])
        for event in events:
            writer.writerow(event)

# Audio Events #######################################################################################################
######################################################################################################################


def save_spectogram_plot(spectogram_matrix, sample_rate, step_size, sample_id, title = 'Spectrogram', x_label = None, y_labels = None, y_tick_num = 6, audio_events = []):
    f = plt.figure(figsize=(10,5), dpi= 80)
    ax = f.add_subplot()
    ax.set_title(title) #set title 
    spectogram = ax.matshow(spectogram_matrix, aspect="auto", cmap=plt.get_cmap('magma')) #draw matrix with colormap 'magma'

    if audio_events:
        colors = cm.rainbow(np.linspace(0, 1, len(audio_events)))
        for index, event in enumerate(audio_events):
            ax.add_patch(patches.Rectangle((event[0], event[2]), event[1] - event[0], event[3] - event[2], linewidth=1, edgecolor=colors[index], facecolor='none'))


    ax.xaxis.set_ticks_position('bottom') #set x ticks to bottom of graph 
    ax.set_xlabel('Time (sec)')
    locator_num = 16 if spectogram_matrix.shape[1] * step_size // sample_rate >= 16 else spectogram_matrix.shape[1] * step_size // sample_rate
    ax.set_xlim(left = 0, right = spectogram_matrix.shape[1])
    ax.xaxis.set_major_locator(ticker.LinearLocator(locator_num))
    formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%-S', time.gmtime(ms * step_size // sample_rate)))
    ax.xaxis.set_major_formatter(formatter)
    
    ax.invert_yaxis() #have y axis start from the bottom
    ax.set_ylabel('Hz')
    y_tick_steps = int(len(y_labels) / y_tick_num)
    ax.set_yticks(np.arange(0, len(y_labels), y_tick_steps))
    ax.set_yticklabels(y_labels[0::y_tick_steps])

    plt.tight_layout()
    plt.colorbar(spectogram, format='%+2.0f dB')
    plt.savefig(os.path.join(plots_path, str(sample_id), title + '.png'))
    #plt.show()

def create_spectogram(sample, window_size, step_size, main_sample_rate):
    #create time series frames
    time_series_frames = get_even_time_series_frames(sample, window_size, step_size)
    #apply taper function to each frame
    apply_hanning_window(time_series_frames)
    #calculate the power spectrum
    power_frames = calculate_power_spectrum(time_series_frames, window_size)
    #get the frequency bins from for the window size
    rfft_bin_freq = get_frequency_bins(window_size, main_sample_rate)
    #convert power spectogram to decibel units
    db_frames = librosa.power_to_db(power_frames, ref=np.max)

    return db_frames, rfft_bin_freq

def process_spectogram(db_frames):
    #test #######
    relevant_freq_range = 21 #always use uneven number
    time_dim = db_frames.shape[0]
    freq_dim = db_frames.shape[1]
    freq_padding = int(relevant_freq_range/2)
    accumulated_frames = np.zeros((time_dim, freq_dim - freq_padding * 2))

    for time_index in range(time_dim):
        for freq_index in range(freq_padding, freq_dim - freq_padding):
            accumulated_frames[time_index][freq_index - freq_padding] = np.median(db_frames[time_index][freq_index - freq_padding : freq_index + freq_padding])
    #test #######

    return accumulated_frames

def detect_events_in_spectogram(spectrogram, event_detection_parameters):
    spectogram_peaks = auxiliary_z_score_peak_detection_2D(spectrogram, event_detection_parameters.mean_lag_window_size, event_detection_parameters.std_lag_window_size, event_detection_parameters.mean_influence, event_detection_parameters.std_influence, event_detection_parameters.threshold)
    #remove negative peaks
    return np.clip(spectogram_peaks, 0, 1)

def cluster_audio_events(spectogram_peaks):
    #get peak coordinates
    peak_coordinates = np.array(np.where(spectogram_peaks == 1)).T
    #group single peaks into audio events
    audio_event_index_scopes = get_audio_event_scopes_from_hdbscan(peak_coordinates, additional_parameters.generate_process_plots, sample_id)
    #sort audio events by their occurrence 
    audio_event_index_scopes = sorted(audio_event_index_scopes, key = lambda x: x[1])
    #process audio events
    return concatenate_events(audio_event_index_scopes, 0)

def create_audio_events_from_sample(process_function, sample_id, 
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

def create_audio_events_with_zscore(
    sample_id, 
    audio_conversion_parameters, 
    event_detection_parameters, 
    clustering_parameters, 
    event_processing_parameters, 
    additional_parameters ):

    #load sample as time series array
    sample, _ = load_audio_sample(sample_id, audio_conversion_parameters.main_sample_rate)

    db_frames, rfft_bin_freq = create_spectogram(sample, audio_conversion_parameters.window_size, audio_conversion_parameters.step_size, audio_conversion_parameters.main_sample_rate)
    #trim the spectogram to a specified frequency range
    db_frames, rfft_bin_freq = trim_to_frequency_range(db_frames, rfft_bin_freq, audio_conversion_parameters.max_frequency, audio_conversion_parameters.min_frequency)

    db_frames = process_spectogram(db_frames)    

    transposed_db_spectogram = np.array(db_frames).T #transpose spectogram frames dimension from (time_step, frequency) to (frequency, time_step)

    if additional_parameters.generate_process_plots:    #save the decibel spectogram
        save_spectogram_plot(transposed_db_spectogram, audio_conversion_parameters.main_sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram (Decibel)", y_labels = rfft_bin_freq)

    #detect peaks in spectogram
    spectogram_peaks = detect_events_in_spectogram(transposed_db_spectogram, event_detection_parameters)

    if additional_parameters.generate_process_plots: #save_spectogram_peaks()
        save_spectogram_plot(spectogram_peaks, audio_conversion_parameters.main_sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram Peaks", y_labels = rfft_bin_freq)

    #accumulate individual audio peaks to sound events
    audio_event_index_scopes = cluster_audio_events(spectogram_peaks)

    if additional_parameters.generate_process_plots:
        save_spectogram_plot(transposed_db_spectogram, audio_conversion_parameters.main_sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram with Events", y_labels = rfft_bin_freq, audio_events=audio_event_index_scopes)
    #convert audio event index scopes to audio events with time and frequency scopes
    audio_events = index_scopes_to_unit_scopes(audio_event_index_scopes, audio_conversion_parameters.main_sample_rate, step_size, rfft_bin_freq)
    #save events
    return audio_events


if __name__ == '__main__':
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
    args = parser.parse_args()

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

    additional_parameters = AdditionalParameters(args.generate_process_plots)
    
    if args.multi_processing:
        print(f"Using {args.multi_processing} cores for multi processing!")
        with multiprocessing.Pool(processes=args.multi_processing) as pool:
            pool.starmap(create_audio_events_from_sample, [(create_audio_events_with_zscore, sample_id, audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters) for sample_id in sample_id_arr])
    else:
        for sample_id in sample_id_arr:
            create_audio_events_from_sample(create_audio_events_with_zscore, sample_id, audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters)


