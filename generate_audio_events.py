import audio_processing as ap

import multiprocessing
import argparse
import os
from collections import namedtuple

audio_events_dir = ap.audio_events_dir
plots_path = ap.plots_path
raw_sample_dir = ap.raw_sample_dir

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
event_distance_max = 0#int(0.1 * main_sample_rate) # 100ms * sample rate for the max audio event distance
event_freq_differnce_max = 0
event_length_min = int(0.125 * main_sample_rate) # 125ms * sample rate will give us the max distance in "index" units

#Events to audio variables
#set half a sec as buffer lengths
start_buffer_len = int(.5 * main_sample_rate)
end_buffer_len = int(.5 * main_sample_rate)

audio_conversion_parameters = ap.AudioConversionParameters(main_sample_rate, window_size, step_size, max_frequency, min_frequency)
event_detection_parameters = ap.EventDetectionParameters(mean_lag_window_size, std_lag_window_size, mean_influence, std_influence, threshold)
clustering_parameters = ap.ClusteringParameters(min_cluster_size)
event_processing_parameters = ap.EventProcessingParameters(event_distance_max, event_freq_differnce_max, event_length_min, start_buffer_len, end_buffer_len)

if __name__ == "__main__":
    ap.main(audio_conversion_parameters, event_detection_parameters, clustering_parameters, event_processing_parameters, ap.create_audio_events_with_zscore)