#%%
import numba as nb
import numpy as np
import audio_processing as ap 
import time
from collections import namedtuple


#%%
def process_spectrogram(db_frames, rfft_bin_freq, audio_processing_parameters):
    return db_frames, rfft_bin_freq
#%%
def standard_function():
    #Variables for audio detection
    #Variables for audio detection
    sample_rate = 44100 # 44.1 kHz
    ##Spectogram variables
    window_size = 2048
    step_size = 512

    ##Frequency trims variables
    max_frequency = 10000
    min_frequency = 1000

    #relevant_freq_range
    relevant_freq_range = 5

    ##z score peak detection variables
    mean_lag_window_size = 256
    std_lag_window_size = 256
    mean_influence = .025#.001
    std_influence = .025#001
    threshold = 2.25#2.5

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

    CustomAudioConversionParameters, CustomAudioProcessingParameters, CustomEventDetectionParameters, CustomClusteringParameters, CustomEventProcessingParameters, AdditionalParameters = ap.getCustomParameters()

    audio_conversion_parameters = CustomAudioConversionParameters(sample_rate, window_size, step_size, max_frequency, min_frequency)
    audio_processing_parameters = CustomAudioProcessingParameters(process_spectrogram, relevant_freq_range)
    event_detection_parameters = CustomEventDetectionParameters(ap.detect_peaks_in_spectrogram, mean_lag_window_size, std_lag_window_size, mean_influence, std_influence, threshold)
    clustering_parameters = CustomClusteringParameters(ap.cluster_audio_events, min_cluster_size)
    event_processing_parameters = CustomEventProcessingParameters(event_distance_max, event_freq_differnce_max, event_length_min, start_buffer_len, end_buffer_len)
    additional_parameters = AdditionalParameters(True)

    ap.detect_audio_events_with_custom(2243804495, audio_conversion_parameters, audio_processing_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters)
#%%

# start_time = time.time()
# standard_function()
# end_time = time.time()
# print(f"standard fn runtime: {end_time-start_time}s")

#%%
def detect_peaks_in_spectrogram(spectrogram, event_detection_parameters):
    spectrogram_peaks = numba_zscore_detect_peaks(spectrogram, event_detection_parameters.mean_lag_window_size, event_detection_parameters.std_lag_window_size, event_detection_parameters.mean_influence, event_detection_parameters.std_influence, event_detection_parameters.threshold)

    return np.clip(spectrogram_peaks, 0, 1)
    
# %%
nb.jit('float64[:,:](float64[:,:], int, int, float32, float32, float32)', nopython=True, parallel=True)
def numba_zscore_detect_peaks(spectrogram : np.array, mean_lag : int, std_lag : int, mean_influence : float, std_influence : float, threshold : float):
    spectrogram_peaks_matrix = np.zeros(spectrogram.shape) 
    max_lag_length = mean_lag if mean_lag > std_lag else std_lag #fing the larger lag
    #get the windows for mean and std 
    mean_window_matrix = np.asarray(spectrogram[:,max_lag_length - mean_lag : max_lag_length])
    std_window_matrix = np.asarray(spectrogram[:,max_lag_length - std_lag : max_lag_length])
    
    for freq_index in nb.prange(spectrogram.shape[0]):
        lag_window_mean = np.mean(mean_window_matrix[freq_index])
        lag_window_std = np.std(std_window_matrix[freq_index])

        for time_index in range(max_lag_length, spectrogram.shape[1]):
            mean_window_matrix[freq_index] = np.roll(mean_window_matrix[freq_index], -1)
            std_window_matrix[freq_index] = np.roll(std_window_matrix[freq_index], -1)

            if abs(spectrogram[freq_index][time_index] - lag_window_mean) > threshold * lag_window_std:
                if spectrogram[freq_index][time_index] > lag_window_mean: #check for positive or negative peak
                    spectrogram_peaks_matrix[freq_index][time_index] = 1
                else:
                    spectrogram_peaks_matrix[freq_index][time_index] = -1

                mean_window_matrix[freq_index][-1] = mean_influence * spectrogram[freq_index][time_index] + (1 - mean_influence) * mean_window_matrix[freq_index][-2]            
                std_window_matrix[freq_index][-1] = std_influence * spectrogram[freq_index][time_index] + (1 - std_influence) * std_window_matrix[freq_index][-2]            
            else:
                mean_window_matrix[freq_index][-1] = spectrogram[freq_index][time_index]            
                std_window_matrix[freq_index][-1] = spectrogram[freq_index][time_index]

            lag_window_mean = np.mean(mean_window_matrix[freq_index])            

    return spectrogram_peaks_matrix

#%%
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

# %%
nb.jit('float64[:,:](float64[:,:], int, int, float32, float32, float32)', nopython=True, parallel=True)
def numba_zscore_detect_peaks_2(frequency_time_series : np.array, mean_lag : int, std_lag : int, mean_influence : float, std_influence : float, threshold : float):
    signal_peaks_matrix = np.zeros(frequency_time_series.shape)
    
    for freq_index in nb.prange(frequency_time_series.shape[0]):
        signal_peaks_matrix[freq_index] = auxiliary_z_peak_detection(frequency_time_series[freq_index,:], mean_lag, std_lag, mean_influence, std_influence, threshold)           

    return signal_peaks_matrix


#%%
def numba_function():
    #Variables for audio detection
    #Variables for audio detection
    sample_rate = 44100 # 44.1 kHz
    ##Spectogram variables
    window_size = 2048
    step_size = 512

    ##Frequency trims variables
    max_frequency = 10000
    min_frequency = 1000

    #relevant_freq_range
    relevant_freq_range = 5

    #
    chunk_size_sec = 5

    ##z score peak detection variables
    mean_lag_window_size = 256
    std_lag_window_size = 256
    mean_influence = .025#.001
    std_influence = .025#001
    threshold = 2.25#2.5

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

    CustomAudioConversionParameters, CustomAudioProcessingParameters, CustomEventDetectionParameters, CustomClusteringParameters, CustomEventProcessingParameters, AdditionalParameters = ap.getCustomParameters()

    CustomExtAudioConversionParameters = namedtuple('CustomAudioConversionParameters', [ 'sample_rate', 'window_size', 'step_size', 'max_frequency', 'min_frequency', 'chunk_len_sec'])

    audio_conversion_parameters = CustomExtAudioConversionParameters(sample_rate, window_size, step_size, max_frequency, min_frequency, chunk_size_sec)
    # audio_processing_parameters = CustomAudioProcessingParameters(process_spectrogram, relevant_freq_range)
    event_detection_parameters = CustomEventDetectionParameters(detect_peaks_in_spectrogram, mean_lag_window_size, std_lag_window_size, mean_influence, std_influence, threshold)
    clustering_parameters = CustomClusteringParameters(ap.cluster_audio_events, min_cluster_size)
    event_processing_parameters = CustomEventProcessingParameters(event_distance_max, event_freq_differnce_max, event_length_min, start_buffer_len, end_buffer_len)
    additional_parameters = AdditionalParameters(True)
    #2243804495
    ap.detect_audio_events_with_zscore_chunked(2243890931, audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, additional_parameters)

#%%
start_time = time.time()
numba_function()
end_time = time.time()
print(f"chunked fn runtime: {end_time-start_time}s")
# %%
