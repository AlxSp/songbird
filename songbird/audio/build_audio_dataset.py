#%%
from typing import List
from songbird.dataset.dataset_info import DatasetInfo, SampleRecordingType
import songbird.audio.audio_processing as ap

import torch
import torchaudio

import os
import json
import shutil
import random

import numpy as np
import scipy as sp
from scipy import signal
from scipy.ndimage.filters import maximum_filter

from datetime import datetime
from dataclasses import asdict, dataclass, field
import tqdm
from multiprocessing import Pool

#%%
@dataclass 
class SpectrogramEvent:
    start_time_index: int
    end_time_index: int
    min_frequency_index: int
    max_frequency_index: int
    
@dataclass
class TimeSeriesEvent:
    start_time_sec: float
    end_time_sec: float
    min_frequency_hz: float
    max_frequency_hz: float

def get_parent_dir(path, num_levels = 1):
    for _ in range(num_levels):
        path = os.path.dirname(path)        
    return path

def set_random_seed(random_seed):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def get_threshold_mask(spectrogram, std_threshold):
    maximums = maximum_filter(spectrogram, size=20)
    mask = maximums > np.mean(maximums) + std_threshold * np.std(maximums)
    return mask

def find_objects_in_mask(mask):
    labeled_mask, num_of_events = sp.ndimage.label(mask)
    audio_event_data_slices = sp.ndimage.find_objects(labeled_mask)
    return audio_event_data_slices

def spectrogram_indices_to_time_series_indices(start_window, end_window, stft_window_size, stft_step_size):
    start_time = start_window * stft_step_size
    end_time = end_window * stft_step_size + stft_window_size
    return start_time, end_time

def spectrogram_indices_to_seconds(start_window, end_window, stft_window_size, stft_step_size, sample_rate):
    start_time = start_window * stft_step_size / sample_rate
    end_time = (end_window * stft_step_size + stft_window_size) / sample_rate
    return start_time, end_time

def project_events_to_original_sample(events, cropped_frequency_bins, frequency_bins):
    projected_events = []
    for time_start_index, time_end_index, freq_start_index, freq_end_index in events:
        start_freq = cropped_frequency_bins[freq_start_index]
        end_freq = cropped_frequency_bins[freq_end_index - 1]
        projected_freq_start_index = np.argmin(np.abs(frequency_bins - start_freq))
        projected_freq_end_index = np.argmin(np.abs(frequency_bins - end_freq))
        projected_events.append((time_start_index, time_end_index, projected_freq_start_index, projected_freq_end_index))

    return projected_events

def create_spectrogram(
        sample, 
        sample_rate, 
        stft_window_size, 
        stft_step_size, 
        ):

    spectrogram_transform = torchaudio.transforms.Spectrogram(
      n_fft = stft_window_size,
      win_length = stft_window_size,
      hop_length = stft_step_size,
      center = True,
      pad_mode = "reflect",
      power = 2.0,
    ).to(device)
    # create spectrogram
    spectrogram = spectrogram_transform(torch.from_numpy(sample).to(device))
    spectrogram =  torchaudio.transforms.AmplitudeToDB('power', top_db=80)(spectrogram)
    spectrogram = spectrogram.to("cpu").numpy()
    # get frquency bins    
    frequency_bins = torch.fft.rfftfreq(stft_window_size, d = 1.0/sample_rate).numpy()
    
    return spectrogram, frequency_bins

def crop_spectrogram_to_frequency_range(spectrogram, frequency_bins, frequency_range_size, lower_frequency_margin):
    freq_max_index = frequency_range_size + lower_frequency_margin#np.argmin(np.abs(frequency_bins - max_frequency))
    freq_min_index =  lower_frequency_margin
    cropped_spectrogram, cropped_frequency_bins = spectrogram[freq_min_index:freq_max_index,:], frequency_bins[freq_min_index:freq_max_index]
    
    return cropped_spectrogram, cropped_frequency_bins

def get_events_from_spectrogram_mask(spectrogram_mask):
    spectrogram_event_data_slices = find_objects_in_mask(spectrogram_mask)
    # convert audio events to spectrogram events
    return [[time_slice.start, time_slice.stop, frequency_slice.start, frequency_slice.stop] for frequency_slice, time_slice in spectrogram_event_data_slices]

def get_events_from_spectrogram(
        sample, 
        sample_rate, 
        stft_window_size, 
        stft_step_size, 
        frequency_range_size,
        lower_frequency_margin,
        std_threshold = 1.2,
        device = "cpu"
        ):
    
    spectrogram_transform = torchaudio.transforms.Spectrogram(
      n_fft = stft_window_size,
      win_length = stft_window_size,
      hop_length = stft_step_size,
      center = True,
      pad_mode = "reflect",
      power = 2.0,
    ).to(device)
    # create spectrogram
    spectrogram = spectrogram_transform(torch.from_numpy(sample).to(device))
    spectrogram =  torchaudio.transforms.AmplitudeToDB('power', top_db=80)(spectrogram)
    spectrogram = spectrogram.to("cpu").numpy()
    # get frquency bins    
    frequency_bins = torch.fft.rfftfreq(stft_window_size, d = 1.0/sample_rate).numpy()

    # crop spectrogram frequencies
    freq_max_index = frequency_range_size + lower_frequency_margin#np.argmin(np.abs(frequency_bins - max_frequency))
    freq_min_index =  lower_frequency_margin
    cropped_spectrogram, frequency_bins =  spectrogram[freq_min_index:freq_max_index,:], frequency_bins[freq_min_index:freq_max_index]
    
    # get threshold mask
    mask = get_threshold_mask(cropped_spectrogram, std_threshold)
    
    # detect audio events from mask and return audio events
    spectrogram_event_data_slices = find_objects_in_mask(mask)
    # convert audio events to spectrogram events
    spectrogram_events_arr = [[time_slice.start, time_slice.stop, frequency_slice.start, frequency_slice.stop] for frequency_slice, time_slice in spectrogram_event_data_slices]
    
    return spectrogram_events_arr, spectrogram.shape

def get_events_from_time_series():
    # TODO: implement
    raise NotImplementedError

def convert_events_to_sample_slice_indices_array(events_arr, sampling_window_size, sampling_step_size, sampling_padding_size, data_size):
    """
    Converts a list of events into a list of event slice indices given the sampling arguments.

    ----------
    Parameters:
        events_arr: List[]
            List of events to convert.
        sampling_window_size: int
            Size of the window to sample from.
        sampling_step_size: int
            Size of the step to sample with.
        sampling_padding_size: int
            Size of the padding to add to the start and end of the window.
        data_size: int
            Size of the data to sample from.

    ----------
    Returns:
        sample_slice_indices_arr: np.array[[int, int]]
            Array of event slice indices sorted by start index.
    """
    start_indices_arr = []
    end_indices_arr = []

    for event_start_index, event_end_index in events_arr: # audio events are in spectrogram coordinates
        event_length = event_end_index - event_start_index    #length of the event

        if event_length < sampling_window_size: #if the sample time dimension is larger than the length of the event
            #adjust the length of the event to fit the sample time dimension
            event_middle_index = event_length // 2 + event_start_index   #index of the middle of the event
            event_end_index = event_middle_index + sampling_window_size // 2 #adjust the end index of the event to fit the sample time dimension
            event_start_index = event_end_index - sampling_window_size  #adjust the start index of the event to fit the sample time dimension
        
        #index of the start of the event with sample padding. Set to zero if index is negative
        sample_range_start_index = max(event_start_index - sampling_padding_size, 0) 
        #index of the end of the event with sample padding. Set to spectrogram length if index is greater than spectrogram length
        sample_range_end_index = min(event_end_index + sampling_padding_size, data_size) 

        start_indices_arr.append(np.arange(sample_range_start_index, sample_range_end_index - sampling_window_size, sampling_step_size)) # get all start indices for the event
        end_indices_arr.append(np.arange(sample_range_start_index + sampling_window_size, sample_range_end_index, sampling_step_size)) # get all end indices for the event

    if not len(start_indices_arr) > 0: #if there are no audio events
        return np.asarray([])
    
    start_indices_arr = np.concatenate(start_indices_arr) #concatenate all the slice indices into a single array
    end_indices_arr = np.concatenate(end_indices_arr) #concatenate all the slice indices into a single array
    sample_slice_indices_arr = np.column_stack((start_indices_arr, end_indices_arr)) #convert the two separate index arrays into a 2d array of [[start, end], ...] 
    
    sample_slice_indices_arr = np.unique(sample_slice_indices_arr, axis = 0) # remove duplicate slices

    # sort the slice indices by start index
    sample_slice_indices_arr = sample_slice_indices_arr[sample_slice_indices_arr[:,0].argsort()]
    
    return np.asarray(sample_slice_indices_arr)
    

def torch_create_audio_samples_from_events(
        sample_id,
        sample_rate,
        stft_window_size,
        stft_step_size,
        frequency_range_size,
        lower_frequency_margin, 
        sampling_window_size, 
        sampling_step_size, 
        sampling_padding_size,
        device = 'cpu'
    ):
    
    # load audio from file
    audio_sample, _ = ap.load_audio_sample(sample_id, sample_rate)
    
    # create high pass filer
    sos = signal.butter(10, 3500, 'hp', fs=sample_rate, output='sos')
    # apply high pass filter
    filtered_audio_sample = signal.sosfilt(sos, audio_sample)
    # create spectrogram
    
    spectrogram_events_arr, spectrogram_shape = get_events_from_spectrogram(filtered_audio_sample, sample_rate, stft_window_size, stft_step_size, frequency_range_size, lower_frequency_margin, device = device)

    audio_events_arr = [spectrogram_indices_to_time_series_indices(start_time_index, end_time_index, stft_window_size, stft_step_size) for start_time_index, end_time_index, _, _ in spectrogram_events_arr]

    sample_slice_indices_arr = convert_events_to_sample_slice_indices_array(audio_events_arr, sampling_window_size, sampling_step_size, sampling_padding_size, len(audio_sample))

    return audio_sample, sample_slice_indices_arr
    
def torch_create_spectrogram_samples(
        sample_id, 
        sample_rate,
        stft_window_size,
        stft_step_size,
        frequency_range_size,
        lower_frequency_margin, 
        sample_dim, 
        sampling_step_size, 
        sampling_padding_size,
        device = 'cpu'
    ):
    #ap.empty_or_create_dir(os.path.join(ap.plots_path, str(sample_id)))
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f'Device: {device}')
    #start_time = process_time()
    sample, _ = ap.load_audio_sample(sample_id, sample_rate)
    
    normalized_sample = (sample - np.mean(sample)) / np.std(sample)#min_max_normalize(sample)
    #print(f"Task: {'Load sample':<28} | Time taken: {process_time() - start_time:>9.3f} sec | Sample length: {len(sample) / sample_rate:.3f} sec")
    # create high pass filter
    #start_time = process_time()
    sos = signal.butter(10, 3500, 'hp', fs=sample_rate, output='sos')
    # apply high pass filter
    filtered_sample = signal.sosfilt(sos, sample)
    # print(f"Task: {'Apply High Pass Filter':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")
    # create spectrogram from audio sample
    # start_time = process_time()
    spectrogram_transform = torchaudio.transforms.Spectrogram(
      n_fft = stft_window_size,
      win_length = stft_window_size,
      hop_length = stft_step_size,
      center = True,
      pad_mode = "reflect",
      power = 2.0,
    ).to(device)
    spectrogram = spectrogram_transform(torch.from_numpy(filtered_sample).to(device))
    spectrogram =  torchaudio.transforms.AmplitudeToDB('power', top_db=80)(spectrogram)
    frequency_bins = torch.fft.rfftfreq(stft_window_size, d = 1.0/sample_rate).numpy()
    spectrogram = spectrogram.to("cpu").numpy()

    # print(f"spectrogram shape: {spectrogram.shape}")
    # print(f"Task: {'Create Spectrogram':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")
    # ap.save_spectrogram_plot(spectrogram, sample_rate, stft_step_size, sample_id, title = "Pytorch Spectogram (Decibel)", y_labels = frequency_bins)
    # crop spectrogram's frequency range
    # start_time = process_time()

    #frequency_range_size = sample_dim[1] 
    #lower_frequency_margin = 100

    freq_max_index = frequency_range_size + lower_frequency_margin#np.argmin(np.abs(frequency_bins - max_frequency))
    freq_min_index =  lower_frequency_margin
    cropped_spectrogram, frequency_bins =  spectrogram[freq_min_index:freq_max_index,:], frequency_bins[freq_min_index:freq_max_index]
    #cropped_spectrogram, frequency_bins = trim_to_frequency_range(spectrogram, frequency_bins, img_dim[1], 100)
    # print(f"Task: {'Crop Spectrogram':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")

    # ap.save_spectrogram_plot(cropped_spectrogram, sample_rate, stft_step_size, sample_id, title = "Pytorch Cropped Spectogram (Decibel)", y_labels = frequency_bins)

    # start_time = process_time()
    std_threshold = 1.2
    mask = get_threshold_mask(cropped_spectrogram, std_threshold)
    # print(f"Task: {'Create Mask':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")
    
    # ap.save_spectrogram_plot(cropped_spectrogram * mask, sample_rate, stft_step_size, sample_id, title = "Pytorch Masked Spectogram (Decibel)", y_labels = frequency_bins)

    # start_time = process_time()
    audio_event_data_slices = get_events_from_spectrogram(mask)
    audio_events_arr = [SpectrogramEvent(time_slice.start, time_slice.stop, frequency_slice.start, frequency_slice.stop) for frequency_slice, time_slice in audio_event_data_slices]
    # print(f"Task: {'Get Audio Events':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")

    # start_time = process_time()

    sample_time_dim = sample_dim[1]
    start_indices_arr = []
    end_indices_arr = []
    # start_time_0 = process_time()
    for audio_event in audio_events_arr:

        event_start_index = audio_event.start_time_index
        event_end_index = audio_event.end_time_index
        event_length = event_end_index - event_start_index    #length of the event

        if event_length < sample_time_dim: #if the sample time dimension is larger than the length of the event
            #adjust the length of the event to fit the sample time dimension
            event_middle_index = event_length // 2 + event_start_index   #index of the middle of the event
            event_end_index = event_middle_index + sample_time_dim // 2 #adjust the end index of the event to fit the sample time dimension
            event_start_index = event_end_index - sample_time_dim  #adjust the start index of the event to fit the sample time dimension
        
        #index of the start of the event with sample padding. Set to zero if index is negative
        sample_range_start_index = max(event_start_index - sampling_padding_size, 0) 
        #index of the end of the event with sample padding. Set to spectrogram length if index is greater than spectrogram length
        sample_range_end_index = min(event_end_index + sampling_padding_size, spectrogram.shape[1]) 

        start_indices_arr.append(np.arange(sample_range_start_index, sample_range_end_index - sample_time_dim, sampling_step_size)) # get all start indices for the event
        end_indices_arr.append(np.arange(sample_range_start_index + sample_time_dim, sample_range_end_index, sampling_step_size)) # get all end indices for the event
    # print(f"Task: {'Get Slice Indices':<28} | Time taken: {process_time() - start_time_0:>9.3f} sec |")

    audio_events_found = len(start_indices_arr) > 0
    # start_time_1 = process_time()
    if not audio_events_found: #if there are no audio events
        return [], [], audio_events_found
    
    

    start_indices_arr = np.concatenate(start_indices_arr) #concatenate all the slice indices into a single array
    end_indices_arr = np.concatenate(end_indices_arr) #concatenate all the slice indices into a single array
    sample_slice_indices_arr = np.column_stack((start_indices_arr, end_indices_arr)) #convert the two separate index arrays into a 2d array of [[start, end], ...] 
    
    sample_slice_indices_arr = np.unique(sample_slice_indices_arr, axis = 0) # remove duplicate slices
    
    # convert spectrogram indices to time indices

    spectrogram_sample_arr = np.array([cropped_spectrogram[:, start_index:end_index] for start_index, end_index in sample_slice_indices_arr])#cropped_spectrogram[:,sample_slice_indices_arr[:,0]:sample_slice_indices_arr[:,1]] # get the spectrogram slices
    # print(f"Task: {'Get Samples From Spectrogram':<28} | Time taken: {process_time() - start_time_1:>9.3f} sec |")


    # print(f"Task: {'Create samples':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")
    return spectrogram_sample_arr, sample_slice_indices_arr, audio_events_found

def create_sequential_train_test_indices(sample_slice_range_arr, test_split = 0.1):
    sample_count = sample_slice_range_arr.shape[0]

    all_indices = np.arange(sample_count)
    test_sample_count = int(sample_count * test_split) # number of test samples

    test_indices_arr = all_indices[-test_sample_count:] # get the test indices
    test_start_slice_start, _ = sample_slice_range_arr[test_indices_arr[0]] # get the start index of the test samples
    train_indices_arr = all_indices[(sample_slice_range_arr[:,1] - test_start_slice_start <= 0)] # get the train indices. 

    # print(sample_count - len(test_indices_arr) - len(train_indices_arr)) # number of samples that are excluded from test and train

    if np.any(np.in1d(train_indices_arr, test_indices_arr)): #make sure the train and test indices are not the same
        raise ValueError(f"Train and test indices are overlapping. Number of indices that appear in both sets `{np.sum(train_indices_arr == test_indices_arr)}`")

    return train_indices_arr, test_indices_arr


def create_random_train_test_indices(sample_slice_indices_arr, sampling_window_size, sampling_step_size, sampling_padding_size, test_split = 0.1):
    sample_count = sample_slice_indices_arr.shape[0]
    overlapping_sample_num = sampling_window_size // sampling_step_size # number of following samples that share overlapping data with a sample 
    additional_samples_to_remove_per_test_sample = overlapping_sample_num * 2 # since both the preceding and following samples are overlapping the number of samples to remove is twice the number of 'overlapping' samples
    test_samples_num = int(sample_count / ((1-test_split)/test_split + 1 + additional_samples_to_remove_per_test_sample)) # minimum number of test samples. Since the the overlapping samples will be removed from the train set, the overlapping samples have to be accounted for in the test size calculation
    all_indices = np.arange(sample_count)
    
    if test_samples_num == 0:
        return all_indices, np.array([])
    # TODO: the test_samples_num could be better calculated by taking into account consecutive indices and sampling_padding_size
    consecutive_indices_arr = np.split(all_indices, np.where(np.diff(sample_slice_indices_arr[:,0]) != sampling_step_size)[0] + 1) #split the indices into consecutive indices
    # we don't want to include samples created in the padding area of an event as they may not have useful data regarding the event. Thus we slice each consecutive index array to remove the padding area. On the end of the array we remove the padding size plus the number of samples that 
    # will be removed when taking a one test sample. This also prevents taking a section that overlaps two events.
    consecutive_testable_indices_arr = []
    for consecutive_indices in consecutive_indices_arr:
        indices = consecutive_indices[sampling_padding_size:-(sampling_padding_size + additional_samples_to_remove_per_test_sample + 1)]
        if indices.size != 0:
            consecutive_testable_indices_arr += [[index, index + additional_samples_to_remove_per_test_sample + 1] for index in indices]
        else:
            indices = consecutive_indices[:-(additional_samples_to_remove_per_test_sample + 1)]
            if indices.size != 0:
                consecutive_testable_indices_arr += [[index, index + additional_samples_to_remove_per_test_sample + 1] for index in indices]
            else:
                consecutive_testable_indices_arr += [[consecutive_indices[0], consecutive_indices[-1]]]

    if not consecutive_testable_indices_arr:
        raise ValueError("No valid indices found")
            
    #consecutive_non_padded_indices_arr = np.asarray([index for consecutive_indices in consecutive_indices_arr for index in consecutive_indices[sampling_padding_size:-(sampling_padding_size + additional_samples_to_remove_per_test_sample + 1)] if sampling_padding_size + additional_samples_to_remove_per_test_sample + 1 < len(consecutive_indices) else consecutive_indices[sampling_padding_size:-(sampling_padding_size + additional_samples_to_remove_per_test_sample + 1)]]) #get the non padded indices. 
    random_indices = np.random.choice(len(consecutive_testable_indices_arr), test_samples_num, replace = False) #get random indices to create test samples
    #test_indices_arr = np.random.choice(consecutive_testable_indices_arr, test_samples_num, replace = False) #get the test indices
    test_indices_arr = np.asarray(consecutive_testable_indices_arr)[random_indices]
    # TODO: initialize indices_to_remove_from_train_set with numpy array since the shape is already known
    indices_to_remove_from_train_set = np.concatenate([np.arange(remove_index_range[0], remove_index_range[1]) for remove_index_range in test_indices_arr]) #get the indices to remove from the train set
    train_selection = np.ones(sample_count, dtype = bool)
    train_selection[indices_to_remove_from_train_set] = False #remove the indices from the train set

    train_indices_arr = all_indices[train_selection] #get the train indices
    test_indices_arr = test_indices_arr[:,0] + overlapping_sample_num * (test_indices_arr[:,1] - test_indices_arr[:,0] == additional_samples_to_remove_per_test_sample + 1) #get the test indices
    
    if np.any(np.in1d(train_indices_arr, test_indices_arr)): #make sure the train and test indices are not the same
        raise ValueError(f"Train and test indices are the same. Number of indices that appear in both sets `{(train_indices_arr == test_indices_arr).sum()}`")
        
    return train_indices_arr, test_indices_arr

def group_subsequent_values(value_array, step_size=1):
    """
    Return list of subsequent lists of numbers from vals which are below or meet the step size.
    
    ----------
    value_array : array-like 
        Ascending ordered array of values to group.
    """
    run = [value_array[0]]
    result = [run]
    for value in value_array[1:]:
        if run[-1] + step_size >= value:
            run.append(value)
        else:
            run = [value]
            result.append(run)

    return result

def compact_audio_data(audio_sample, sample_offsets_arr, sampling_window_size):

    sample_range_group_arr = group_subsequent_values(sample_offsets_arr, sampling_window_size) # sampling window size is used so overlapping samples are grouped together

    compacted_audio_data = []
    compacted_sample_offsets = []
    group_offset = 0
    for group_indices in sample_range_group_arr:
        group_start = group_indices[0]
        group_end = group_indices[-1] + sampling_window_size

        compacted_audio_data.append(audio_sample[group_start:group_end])
        compacted_sample_offsets += [start - group_start + group_offset for start in group_indices]
        
        group_offset += group_end - group_start

    compacted_audio_data = np.concatenate(compacted_audio_data)

    return compacted_audio_data, np.asarray(compacted_sample_offsets)


def pytorch_create_samples_from_audio(
        dataset_path, 
        sample_id, 
        sample_rate, 
        stft_window_size, 
        stft_step_size,  
        frequency_range_size, 
        lower_frequency_margin, 
        sample_dim, 
        sampling_step_size, 
        sampling_padding_size,
        validation_split = 0.0,
        random_seed = 0,
        device = "cpu",
    ):
    
    set_random_seed(random_seed)

    sampling_window_size = sample_dim[0]
    
    audio_sample, sample_slice_range_arr = torch_create_audio_samples_from_events(sample_id, sample_rate, stft_window_size, stft_step_size, frequency_range_size, lower_frequency_margin, sampling_window_size, sampling_step_size, sampling_padding_size, device)
    
    if sample_slice_range_arr.size != 0:
        # sample_max = np.max(audio_sample_arr).astype(np.float32)
        # sample_min = np.min(audio_sample_arr).astype(np.float32)
    
        if validation_split > 0.0:
            train_data_path = os.path.join(dataset_path, 'train', 'data')
            train_samples_file_path = os.path.join(train_data_path, f"{sample_id}.bin")
            
            test_data_path = os.path.join(dataset_path, 'test', 'data')
            test_samples_file_path = os.path.join(test_data_path, f"{sample_id}.bin")

            train_indices, test_indices = create_sequential_train_test_indices(sample_slice_range_arr, test_split = validation_split)

            if train_indices.size != 0:

                train_slice_range_arr = sample_slice_range_arr[train_indices]

                compact_train_audio_data, compact_train_sample_offsets = compact_audio_data(audio_sample, train_slice_range_arr[:,0], sampling_window_size)



                random_index = np.random.randint(0, len(train_indices))
                compact_train_offset = compact_train_sample_offsets[random_index]
                audio_sample_start, audio_sample_end = train_slice_range_arr[random_index]
                assert (compact_train_audio_data[compact_train_offset:compact_train_offset + sampling_window_size] == audio_sample[audio_sample_start:audio_sample_end]).all()
                

                np.asarray(compact_train_audio_data, dtype = np.float32).tofile(train_samples_file_path)
                train_data_info_path = os.path.join(dataset_path, 'train', 'data_info')
                train_data_info_file_path = os.path.join(train_data_info_path, f"{sample_id}.json")
                
                compact_train_sample_offsets = compact_train_sample_offsets.tolist()
                train_spectrogram_sample_info_arr = [{"start_index" : start_index, "end_index" : start_index + sampling_window_size} for start_index in compact_train_sample_offsets]
                
                with open(train_data_info_file_path, 'w') as f:
                    json.dump( #spectrogram_img_info_arr, f)
                        {
                            # "max" : float(sample_max),
                            # "min" : float(sample_min),
                            "sample_offsets" : train_spectrogram_sample_info_arr
                        }, f)
                
            if test_indices.size != 0:
                test_slice_range_arr = sample_slice_range_arr[test_indices]
                
                compact_test_audio_data, compact_test_sample_offsets = compact_audio_data(audio_sample, test_slice_range_arr[:,0], sampling_window_size)

                np.asarray(compact_test_audio_data, dtype = np.float32).tofile(test_samples_file_path)
                            
                test_data_info_path = os.path.join(dataset_path, 'test', 'data_info')
                test_data_info_file_path = os.path.join(test_data_info_path, f"{sample_id}.json")     

                compact_test_sample_offsets = compact_test_sample_offsets.tolist()
                test_spectrogram_sample_info_arr = [{"start_index" : start_index, "end_index" : start_index + sampling_window_size} for start_index in compact_test_sample_offsets]
            
                with open(test_data_info_file_path, 'w') as f:
                    json.dump( #spectrogram_img_info_arr, f)
                        {
                            # "max" : float(sample_max),
                            # "min" : float(sample_min),
                            "sample_offsets" : test_spectrogram_sample_info_arr
                        }, f)
                                
        else:
            data_path = os.path.join(dataset_path, 'data')
            samples_file_path = os.path.join(data_path, f"{sample_id}.bin")
            # save the spectrogram samples
            compacted_audio_data, compact_sample_offsets = compact_audio_data(audio_sample, sample_slice_range_arr[:,0], sampling_window_size)

            np.asarray(compacted_audio_data, dtype = np.float32).tofile(samples_file_path)
            
            data_info_path = os.path.join(dataset_path, 'data_info')
            data_info_file_path = os.path.join(data_info_path, f"{sample_id}.json")
            # calculate the spectrogram max and min values
            compact_sample_offsets = compact_sample_offsets.tolist()
            spectrogram_sample_info_arr = [{"start_index" : start_index, "end_index" : start_index + sampling_window_size} for start_index in compact_sample_offsets]
        
            with open(data_info_file_path, 'w') as f:
                json.dump( #spectrogram_img_info_arr, f)
                    {
                        # "max" : float(sample_max),
                        # "min" : float(sample_min),
                        "sample_offsets" : spectrogram_sample_info_arr
                    }, f)

    else:
        print(f"No audio events found for sample {sample_id}")

def save_dataset_attributes(
    dataset_path,
    start_time,
    end_time,
    samples_count,
    sample_type,
    sample_rate,
    stft_window_size,
    stft_step_size,
    sample_dim,
    sampling_step_size,
    sampling_padding_size,
    frequency_range_size,
    lower_frequency_margin,
    validation_split,
    random_seed,
    num_workers,
    device,
    sample_ids
    ):

    with open(os.path.join(dataset_path, 'dataset_attributes.json'), 'w') as f:
            json.dump({
                "start_date" : start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date" : end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "time_taken_sec" : (end_time - start_time).total_seconds(),
                "samples_count" : samples_count,
                # "max_value" : float(max_value),
                # "min_value" : float(min_value),
                
                "build_parameters" : {
                    "dataset_path" : dataset_path,
                    "sample_type" : sample_type,
                    "sample_rate" : sample_rate,
                    "stft_window_size" : stft_window_size,
                    "stft_step_size" : stft_step_size,
                    "sample_dim" : list(sample_dim), 
                    "sampling_step_size" : sampling_step_size, 
                    "sampling_padding_size" : sampling_padding_size,
                    "frequency_range_size" : frequency_range_size, 
                    "lower_frequency_margin" : lower_frequency_margin, 
                    "validation_split" : validation_split,
                    "random_seed" : random_seed,
                    "num_workers" : num_workers,
                    "device" : device,
                    "sample_ids" : sample_ids,
                }
            }, f, indent=2)

def pytorch_create_and_save_dateset(
        dataset_path, 
        sample_type,
        sample_ids, 
        sample_rate, 
        stft_window_size, 
        stft_step_size, 
        sample_dim, 
        sampling_step_size, 
        sampling_padding_size, 
        frequency_range_size, 
        lower_frequency_margin, 
        validation_split = 0.0,
        random_seed = 0,
        num_workers = 1,
        device = 'cpu'
    ):
    
    # check if dataset directory already exists    
    if os.path.exists(dataset_path): 
        shutil.rmtree(dataset_path) # if so delete it
    # create dataset directory
    os.makedirs(dataset_path)
    # create dataset sub directories
    if validation_split > 0.0:
        train_data_path = os.path.join(dataset_path, 'train', 'data')
        train_data_info_path = os.path.join(dataset_path, 'train', 'data_info')
        
        os.makedirs(train_data_path)
        os.makedirs(train_data_info_path)

        
        test_data_path = os.path.join(dataset_path, 'test', 'data')
        test_data_info_path = os.path.join(dataset_path, 'test', 'data_info')
        
        os.makedirs(test_data_path)
        os.makedirs(test_data_info_path)
        
    else:
        data_path = os.path.join(dataset_path, 'data')
        data_info_path = os.path.join(dataset_path, 'data_info')
        
        os.makedirs(data_path)
        os.makedirs(data_info_path)

    print(f"Processing {len(sample_ids)} samples")
    num_workers = min(num_workers, len(sample_ids)) #set the number of workers to the number of samples if the specified worker number is greater than the number of samples
    print(f"Creating dataset with {num_workers} workers")

    start_time = datetime.now()

    if device == 'cuda' and num_workers > 1:
        print("Currently only one worker is supported when device is set to 'cuda'")
        print("Setting num_workers to 1")
        num_workers = 1
    
    if num_workers > 1: # if the workers are more than 1, create a processing pool
        with Pool(processes=num_workers) as pool:
            worker_inputs = [(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  frequency_range_size, lower_frequency_margin, sample_dim, sampling_step_size, sampling_padding_size, validation_split, random_seed, device) for sample_id in sample_ids]
            pool.starmap(pytorch_create_samples_from_audio, worker_inputs, chunksize=1) #setting chunksize to 1 to due to imbalanced sample processing time 
        
    else: # if only one worker is used, create the dataset sequentially on the main thread
        for sample_id in tqdm.tqdm(sample_ids):
            pytorch_create_samples_from_audio(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  frequency_range_size, lower_frequency_margin, sample_dim, sampling_step_size, sampling_padding_size, validation_split, random_seed, device)


    max_value = 0.0
    min_value = np.inf
    samples_count = 0

    if validation_split > 0.0:
        for data_info_file in os.listdir(train_data_info_path): 
            with open(os.path.join(train_data_info_path, data_info_file), 'r') as f:
                data_info = json.load(f)
                # max_value = max(max_value, data_info['max'])
                # min_value = min(min_value, data_info['min'])
                samples_count += len(data_info['sample_offsets'])
                
        for data_info_file in os.listdir(test_data_info_path): 
            with open(os.path.join(test_data_info_path, data_info_file), 'r') as f:
                data_info = json.load(f)
                # max_value = max(max_value, data_info['max'])
                # min_value = min(min_value, data_info['min'])
                samples_count += len(data_info['sample_offsets'])
        
                
        end_time = datetime.now()
        # time_taken_sec = (end_time - start_time).total_seconds()

        save_dataset_attributes(
            os.path.join(dataset_path, 'train'),
            start_time,
            end_time,
            samples_count,
            sample_type,
            sample_rate,
            stft_window_size,
            stft_step_size,
            sample_dim,
            sampling_step_size,
            sampling_padding_size,
            frequency_range_size,
            lower_frequency_margin,
            validation_split,
            random_seed,
            num_workers,
            device,
            sample_ids
            )

        save_dataset_attributes(
            os.path.join(dataset_path, 'test'),
            start_time,
            end_time,
            samples_count,
            sample_type,
            sample_rate,
            stft_window_size,
            stft_step_size,
            sample_dim,
            sampling_step_size,
            sampling_padding_size,
            frequency_range_size,
            lower_frequency_margin,
            validation_split,
            random_seed,
            num_workers,
            device,
            sample_ids
            )

        # with open(os.path.join(, 'dataset_attributes.json'), 'w') as f:
        #     json.dump({
        #         "start_date" : start_time.strftime("%Y-%m-%d %H:%M:%S"),
        #         "end_date" : end_time.strftime("%Y-%m-%d %H:%M:%S"),
        #         "time_taken_sec" : time_taken_sec,
        #         "samples_count" : samples_count,
        #         # "max_value" : float(max_value),
        #         # "min_value" : float(min_value),
                
        #         "build_parameters" : {
        #             "dataset_path" : dataset_path,
        #             "sample_type" : sample_type,
        #             "sample_rate" : sample_rate,
        #             "stft_window_size" : stft_window_size,
        #             "stft_step_size" : stft_step_size,
        #             "sample_dim" : list(sample_dim), 
        #             "sampling_step_size" : sampling_step_size, 
        #             "sampling_padding_size" : sampling_padding_size,
        #             "frequency_range_size" : frequency_range_size, 
        #             "lower_frequency_margin" : lower_frequency_margin, 
        #             "validation_split" : validation_split,
        #             "random_seed" : random_seed,
        #             "num_workers" : num_workers,
        #             "device" : device,
        #             "sample_ids" : sample_ids,
        #         }
        #     }, f, indent=2)
            
        # with open(os.path.join(os.path.join(dataset_path, 'test'), 'dataset_attributes.json'), 'w') as f:
        #     json.dump({
        #         "start_date" : start_time.strftime("%Y-%m-%d %H:%M:%S"),
        #         "end_date" : end_time.strftime("%Y-%m-%d %H:%M:%S"),
        #         "time_taken_sec" : time_taken_sec,
        #         "samples_count" : samples_count,
        #         # "max_value" : float(max_value),
        #         # "min_value" : float(min_value),
                
        #         "build_parameters" : {
        #             "dataset_path" : dataset_path,
        #             "sample_type" : sample_type,
        #             "sample_type" : sample_type,
        #             "sample_rate" : sample_rate,
        #             "stft_window_size" : stft_window_size,
        #             "stft_step_size" : stft_step_size,
        #             "sample_dim" : list(sample_dim), 
        #             "sampling_step_size" : sampling_step_size, 
        #             "sampling_padding_size" : sampling_padding_size,
        #             "frequency_range_size" : frequency_range_size, 
        #             "lower_frequency_margin" : lower_frequency_margin, 
        #             "validation_split" : validation_split,
        #             "random_seed" : random_seed,
        #             "num_workers" : num_workers,
        #             "device" : device,
        #             "sample_ids" : sample_ids,
        #         }
        #     }, f, indent=2)
            
    else:
        for data_info_file in os.listdir(data_info_path): 
            with open(os.path.join(data_info_path, data_info_file), 'r') as f:
                data_info = json.load(f)
                # max_value = max(max_value, data_info['max'])
                # min_value = min(min_value, data_info['min'])
                samples_count += len(data_info['sample_offsets'])

        end_time = datetime.now()
        # time_taken_sec = (end_time - start_time).total_seconds()

        save_dataset_attributes(
            dataset_path,
            start_time,
            end_time,
            samples_count,
            sample_type,
            sample_rate,
            stft_window_size,
            stft_step_size,
            sample_dim,
            sampling_step_size,
            sampling_padding_size,
            frequency_range_size,
            lower_frequency_margin,
            validation_split,
            random_seed,
            num_workers,
            device,
            sample_ids
            )
        # with open(os.path.join(dataset_path, 'dataset_attributes.json'), 'w') as f:
        #     json.dump({
        #         "start_date" : start_time.strftime("%Y-%m-%d %H:%M:%S"),
        #         "end_date" : end_time.strftime("%Y-%m-%d %H:%M:%S"),
        #         "time_taken_sec" : time_taken_sec,
        #         "samples_count" : samples_count,
        #         # "max_value" : float(max_value),
        #         # "min_value" : float(min_value),
        #         "build_parameters" : {
        #             "dataset_path" : dataset_path,
        #             "sample_rate" : sample_rate,
        #             "stft_window_size" : stft_window_size,
        #             "stft_step_size" : stft_step_size,
        #             "sample_dim" : list(sample_dim), 
        #             "sampling_step_size" : sampling_step_size, 
        #             "sampling_padding_size" : sampling_padding_size,
        #             "frequency_range_size" : frequency_range_size, 
        #             "lower_frequency_margin" : lower_frequency_margin, 
        #             "validation_split" : validation_split,
        #             "random_seed" : random_seed,
        #             "num_workers" : num_workers,
        #             "device" : device,
        #             "sample_ids" : sample_ids, 
        #         }
        #     }, f, indent=2)

if __name__ == "__main__":
    
    @dataclass
    class SpectrogramDatasetParameters:
        sample_type : str = 'spectrogram'
        dataset_path : str = None
        # sample_ids: List[int] = field(default_factory=lambda: [2473663, 9475738, 2490719, 2482593, 2494422, 2493052])
        sample_rate : int = 44100

        stft_window_size: int = 2048
        stft_step_size: int = 512

        sample_dim: tuple = (512, 32) # (freq, time)
        sampling_step_size: int = 1  # (time)
        sampling_padding_size: int  = sample_dim[1] // 2
        frequency_range_size: int  = sample_dim[0]

        lower_frequency_margin: int  = 100
        validation_split: int  = 0.1
        random_seed: int  = 42
        num_workers: int  = 1
        device: str = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @dataclass
    class AudioDatasetParameters:
        sample_type : str = 'audio'
        dataset_path : str = None
        # sample_ids: List[int] = field(default_factory=lambda: [2473663, 9475738, 2490719, 2482593, 2494422, 2493052])
        sample_rate : int = 44100

        stft_window_size: int = 2048
        stft_step_size: int = 512

        sample_dim: tuple = (32768, ) # (time)
        sampling_step_size: int = 512  # (time)
        sampling_padding_size: int  = sample_dim[0] // 2

        frequency_range_size: int  = 512
        lower_frequency_margin: int  = 100
        validation_split: int  = 0.1
        random_seed: int  = 42
        num_workers: int  = 12
        device: str = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    dataset_params = AudioDatasetParameters()
    
    project_dir = get_parent_dir(os.path.abspath(__file__), num_levels = 3)
    
    dataset_info = DatasetInfo()
    
    dataset_info.describe_downloaded_samples()
    
    recording_type = SampleRecordingType.Foreground
    species_ids = [2473663, 9475738, 2490719, 2482593, 2494422, 2493052]
    
    sample_ids = []
    for species_id in species_ids:
        sample_ids += dataset_info.get_downloaded_species_sample_ids(species_id, recording_type)
    
    sample_ids = list(set(sample_ids)) # ensure that the sample ids are unique (no duplicates)
    
    # recording_type = SampleRecordingType.Foreground
    
    now = datetime.utcnow()
    dataset_params.dataset_path = os.path.join(project_dir, 'data', 'processed', dataset_params.sample_type, now.strftime("%Y-%m-%d_%H"))
    
    print(f"Dataset set will be stored at: {dataset_params.dataset_path}")
    print("Building dataset")
    
    pytorch_create_and_save_dateset(sample_ids = sample_ids, **asdict(dataset_params))
    print("Dataset built")
    print(f"Stored in location: {dataset_params.dataset_path}")



# @dataclass
# class AudioDatasetParameters:
#     sample_type : str = 'audio'
#     dataset_path : str = None
#     # sample_ids: List[int] = field(default_factory=lambda: [2473663, 9475738, 2490719, 2482593, 2494422, 2493052])
#     sample_rate : int = 44100

#     stft_window_size: int = 2048
#     stft_step_size: int = 512

#     sample_dim: tuple = (32768, ) # (time)
#     sampling_step_size: int = 512  # (time)
#     sampling_padding_size: int  = sample_dim[0] // 2

#     frequency_range_size: int  = 512
#     lower_frequency_margin: int  = 100
#     validation_split: int  = 0.1
#     random_seed: int  = 42
#     num_workers: int  = 1
#     device: str = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # dataset_params = AudioDatasetParameters()
# #%%
# sample_type : str = 'audio'
# dataset_path : str = None
# # sample_ids: List[int] = field(default_factory=lambda: [2473663, 9475738, 2490719, 2482593, 2494422, 2493052])
# sample_rate : int = 44100

# stft_window_size: int = 2048
# stft_step_size: int = 512

# sample_dim: tuple = (32768, ) # (time)
# sampling_step_size: int = 512  # (time)
# sampling_padding_size: int  = sample_dim[0] // 2

# frequency_range_size: int  = 512
# lower_frequency_margin: int  = 100
# validation_split: int  = 0.1
# random_seed: int  = 42
# num_workers: int  = 1
# device: str = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test_sample_id = '2243735323' #'2243804495' #'2243735323'

# std_threshold = 1.2

# plot = False
# #%%
# audio_sample, _ = ap.load_audio_sample(test_sample_id, sample_rate)

# # create high pass filer
# sos = signal.butter(10, 3500, 'hp', fs=sample_rate, output='sos')
# # apply high pass filter
# filtered_audio_sample = signal.sosfilt(sos, audio_sample)
# # create spectrogram

# #%%
# spectrogram, frequency_bins = create_spectrogram(filtered_audio_sample, sample_rate, stft_window_size, stft_step_size)

# #%%
# if plot: ap.save_spectrogram_plot(spectrogram, sample_rate, stft_step_size, test_sample_id, y_labels=frequency_bins)
# # spectrogram_events_arr, spectrogram_shape = get_events_from_spectrogram(filtered_audio_sample, sample_rate, stft_window_size, stft_step_size, frequency_range_size, lower_frequency_margin, device = device)
# #%%
# # cropped_spectrogram, cropped_frequency_bins = crop_spectrogram_to_frequency_range(spectrogram, frequency_bins, lower_frequency_margin, frequency_range_size)

# # #%%
# # ap.save_spectrogram_plot(cropped_spectrogram, sample_rate, stft_step_size, test_sample_id, y_labels=cropped_frequency_bins)


# #%%
# mask = get_threshold_mask(spectrogram, std_threshold)

# #%%
# if plot: ap.save_spectrogram_plot(mask, sample_rate, stft_step_size, test_sample_id, y_labels=frequency_bins)

# #%%
# spectrogram_events_arr = get_events_from_spectrogram_mask(mask)


# #%%
# if plot: ap.save_spectrogram_plot(spectrogram, sample_rate, stft_step_size, test_sample_id, y_labels=frequency_bins, audio_events=spectrogram_events_arr)

# #%%
# audio_events_arr = [spectrogram_indices_to_time_series_indices(start_time_index, end_time_index, stft_window_size, stft_step_size) for start_time_index, end_time_index, _, _ in spectrogram_events_arr]
# #%%
# sample_slice_range_arr = convert_events_to_sample_slice_indices_array(audio_events_arr, sample_dim[0], sampling_step_size, sampling_padding_size, len(audio_sample))


# #%%
# train_indices, test_indices = create_sequential_train_test_indices(sample_slice_range_arr, sample_dim[0], sampling_step_size, sampling_padding_size, test_split = validation_split)


# #spectrogram_sample_arr, sample_slice_indices_arr = torch_create_audio_samples_from_events(test_sample_id, sample_rate = 44100, stft_window_size= 2048, stft_step_size= 512, frequency_range_size= 512, lower_frequency_margin= 100, sample_dim= (32768, ), sampling_step_size= 512, sampling_padding_size =32768 // 2, device= "cpu")
# audio_sample_arr = np.asarray([audio_sample[start_index:end_index] for start_index, end_index in sample_slice_range_arr])