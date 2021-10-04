#%%
from songbird.dataset.dataset_info import DatasetInfo, SampleRecordingType
import songbird.audio.audio_processing as ap

import torch
import torchaudio

import os, sys
import shutil
import pydub
import time
from time import process_time
from datetime import datetime
import numpy as np
import pandas as pd
import tqdm
import json
from dataclasses import dataclass


import scipy as sp
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from skimage.feature import peak_local_max

from multiprocessing import Pool

#%%

@dataclass 
class SpectrogramDim:
    start_time_index: int
    end_time_index: int


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

def crop_frequency_range(
        spectrogram, 
        frequency_bins,
        max_frequency_index: int, 
        min_frequency_index: int 
        ):
    return spectrogram[:, min_frequency_index:max_frequency_index], frequency_bins[min_frequency_index:max_frequency_index]



def trim_to_frequency_range(spectrogram, frequency_bins, frequency_range = 0, min_frequency_margin = 0):
    #get closest frequency in bin
    freq_max_index = frequency_range + min_frequency_margin
    freq_min_index =  min_frequency_margin

    return spectrogram[:,freq_min_index:freq_max_index], frequency_bins[freq_min_index:freq_max_index]

#%%
def get_threshold_mask(spectrogram, std_threshold):
    #print(spectrogram)
    maximums = maximum_filter(spectrogram, size=20)
    mask = maximums > np.mean(maximums) + std_threshold * np.std(maximums)
    return mask
    return spectrogram * mask

def get_audio_events(mask):
    labeled_mask, num_of_events = sp.ndimage.label(mask)
    audio_event_data_slices = sp.ndimage.find_objects(labeled_mask)
    return audio_event_data_slices

def min_max_normalize(data):
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data

#%%
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
    #print(f"Task: {'Load sample':<28} | Time taken: {process_time() - start_time:>9.3f} sec | Sample length: {len(sample) / sample_rate:.3f} sec")
    # create high pass filter
    #start_time = process_time()
    sos = signal.butter(10, 3500, 'hp', fs=sample_rate, output='sos')
    # apply high pass filter
    sample = signal.sosfilt(sos, sample)
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
    spectrogram = spectrogram_transform(torch.from_numpy(sample).to(device))
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
    audio_event_data_slices = get_audio_events(mask)
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

        start_indices_arr.append(np.arange(sample_range_start_index, sample_range_end_index - sample_time_dim)) # get all start indices for the event
        end_indices_arr.append(np.arange(sample_range_start_index + sample_time_dim, sample_range_end_index)) # get all end indices for the event
    # print(f"Task: {'Get Slice Indices':<28} | Time taken: {process_time() - start_time_0:>9.3f} sec |")

    # start_time_1 = process_time()
    start_indices_arr = np.concatenate(start_indices_arr) #concatenate all the slice indices into a single array
    end_indices_arr = np.concatenate(end_indices_arr) #concatenate all the slice indices into a single array
    sample_slice_indices_arr = np.column_stack((start_indices_arr, end_indices_arr)) #convert the two separate index arrays into a 2d array of [[start, end], ...] 
    
    sample_slice_indices_arr = np.unique(sample_slice_indices_arr, axis = 0) # remove duplicate slices

    spectrogram_sample_arr = np.array([cropped_spectrogram[:, start_index:end_index] for start_index, end_index in sample_slice_indices_arr])#cropped_spectrogram[:,sample_slice_indices_arr[:,0]:sample_slice_indices_arr[:,1]] # get the spectrogram slices
    # print(f"Task: {'Get Samples From Spectrogram':<28} | Time taken: {process_time() - start_time_1:>9.3f} sec |")

    spectrogram_sample_info_arr = [{"start_index" : start_index.item(), "end_index" : end_index.item()} for start_index, end_index in sample_slice_indices_arr]

    # print(f"Task: {'Create samples':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")
    return spectrogram_sample_arr, spectrogram_sample_info_arr


#%%
def create_spectrogram_slices(
        sample_id, 
        sample_rate,
        stft_window_size,
        stft_step_size,
        max_frequency,
        min_frequency, 
        img_dim, 
        img_step_size, 
        event_padding_size,
        plot = False
    ):
    
    if plot:
        ap.empty_or_create_dir(os.path.join(ap.plots_path, str(sample_id)))
    # load audio sample
    start_time = process_time()
    sample, _ = ap.load_audio_sample(sample_id, sample_rate)
    print(f"Task: {'Load sample':<28} | Time taken: {process_time() - start_time:>9.3f} sec | Sample length: {len(sample) / sample_rate:.3f} sec")
    # create high pass filter
    start_time = process_time()
    sos = signal.butter(10, 3500, 'hp', fs=sample_rate, output='sos')
    # apply high pass filter
    sample = signal.sosfilt(sos, sample)
    print(f"Task: {'Apply High Pass Filter':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")
    # create spectrogram from audio sample
    start_time = process_time()
    spectrogram, frequency_bins = ap.create_spectrogram(sample, stft_window_size, stft_step_size, sample_rate)
    # print(f"spectrogram shape: {spectrogram.shape}")
    print(f"Task: {'Create Spectrogram':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")
    
    # crop spectrogram's frequency range
    start_time = process_time()
    cropped_spectrogram, frequency_bins = trim_to_frequency_range(spectrogram, frequency_bins, img_dim[1], 100)
    # convert to numpy
    cropped_spectrogram = cropped_spectrogram.numpy()
    print(f"Task: {'Crop Spectrogram':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")

    start_time = process_time()
    mask = get_threshold_mask(cropped_spectrogram, 1.2)
    print(f"Task: {'Create Mask':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")
    
    start_time = process_time()
    audio_event_data_slices = get_audio_events(mask)
    audio_events_arr = [SpectrogramEvent(time_slice.start, time_slice.stop, frequency_slice.start, frequency_slice.stop) for time_slice, frequency_slice in audio_event_data_slices]
    print(f"Task: {'Get Audio Events':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")

    # audio_events_arr = []
    # for time_slice, frequency_slice in audio_event_data_slices:
    #     audio_events_arr.append(spectrogram_audio_event(time_slice.start, time_slice.stop, frequency_slice.start, frequency_slice.stop))
    start_time = process_time()
    spectrogram_img_arr = []
    spectrogram_img_info_arr = []
    
    for audio_event in audio_events_arr: #iterate over all audio events
        event_length = audio_event.end_time_index - audio_event.start_time_index    #length of the event
        #print(f"Original: {audio_event.start_time_index}, {audio_event.end_time_index}, {event_length}")
        spectrogram_img = np.zeros(img_dim) #create empty image
        if event_length < img_dim[0]:   #if event is shorter than image, place it in the middle of the image
            audio_event_middle_index = (event_length) // 2 + audio_event.start_time_index   #index of the middle of the event
            new_audio_event_start_index = audio_event_middle_index - img_dim[0] // 2
            new_audio_event_end_time_index = audio_event_middle_index + img_dim[0] // 2

            # check if new event start is within the actual spectrogram sample
            if new_audio_event_start_index < 0: # if event index is negative, set it to zero and move the end index by the absolute value of the out bounds index
                out_of_range_index = abs(new_audio_event_start_index)
                new_audio_event_start_index += out_of_range_index
                new_audio_event_end_time_index += out_of_range_index
            # check if new event end is within the actual spectrogram            
            if new_audio_event_end_time_index > cropped_spectrogram.shape[0]:
                new_audio_event_end_time_index = cropped_spectrogram.shape[0]
                new_audio_event_start_index = new_audio_event_end_time_index - img_dim[0] if new_audio_event_end_time_index - img_dim[0] > 0 else 0
            
            new_event_length = new_audio_event_end_time_index - new_audio_event_start_index
            noise_data = cropped_spectrogram[np.where(np.invert(mask))]
            noise_mean = np.mean(noise_data)
            noise_std = np.std(noise_data)

            # print(f"Noise mean {noise_mean} noise std {noise_std}")
            # print(noise_data.shape)
            spectrogram_img = np.random.normal(loc = noise_mean, scale=noise_std, size = img_dim)
            spectrogram_img[: new_event_length, :] = cropped_spectrogram[new_audio_event_start_index: new_audio_event_end_time_index, :]

            spectrogram_img_arr.append(spectrogram_img)
            spectrogram_img_info_arr.append({"start_index" : new_audio_event_start_index, "end_index" : new_audio_event_end_time_index})

        elif event_length > img_dim[0]: #if event is longer than image, slide a window over the spectrogram in the event range
            
            event_start_index = audio_event.start_time_index - event_padding_size   #index of the start of the event with sample padding
            event_start_index = 0 if event_start_index < 0 else event_start_index   #set to zero if index is negative

            event_end_index = audio_event.start_time_index + event_length + event_padding_size  #index of the end of the event with sample padding
            event_end_index = spectrogram.shape[0] if event_end_index > spectrogram.shape[0] else event_end_index   #set to spectrogram length if index is greater than spectrogram length

            for start_index in range(event_start_index, event_end_index, img_step_size):    #slide a window of img_dim size over the spectrogram
                end_index = start_index + img_dim[0] #index of the end of the window
                if end_index > event_end_index: #if the window is out of bounds, set it's to the end of the event
                    noise_data = cropped_spectrogram[np.where(np.invert(mask))] #get the data that is hidden by the mask (expected to be noise)
                    noise_mean = np.mean(noise_data)    #get the mean of the noise data
                    noise_std = np.std(noise_data)  #get the standard deviation of the noise data
                    spectrogram_img = np.random.normal(loc = noise_mean, scale=noise_std, size = img_dim)   #fill the spectrogram image with noise

                    spectrogram_img[0:event_end_index-start_index,:] = cropped_spectrogram[start_index: event_end_index, :] #add the spectrogram event data to the image
                else:
                    spectrogram_img = cropped_spectrogram[start_index: end_index, :]    #add the spectrogram event data to the image

                spectrogram_img_arr.append(spectrogram_img)
                spectrogram_img_info_arr.append({"start_index" : start_index, "end_index" : end_index})

    print(f"Task: {'Create samples':<28} | Time taken: {process_time() - start_time:>9.3f} sec |")

    return cropped_spectrogram * mask , audio_events_arr, spectrogram_img_arr, spectrogram_img_info_arr

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
        device
    ):
    
    spectrogram_sample_arr, spectrogram_img_info_arr = torch_create_spectrogram_samples(sample_id, sample_rate, stft_window_size, stft_step_size, frequency_range_size, lower_frequency_margin, sample_dim, sampling_step_size, sampling_padding_size, device)
    
    data_path = os.path.join(dataset_path, 'data')
    samples_file_path = os.path.join(data_path, f"{sample_id}.bin")
    # save the spectrogram samples
    np.asarray(spectrogram_sample_arr, dtype = np.float32).tofile(samples_file_path)

    
    data_info_path = os.path.join(dataset_path, 'data_info')
    data_info_file_path = os.path.join(data_info_path, f"{sample_id}.json")
    # calculate the spectrogram max and min values
    sample_max = np.max(spectrogram_sample_arr).astype(np.float32)
    sample_min = np.min(spectrogram_sample_arr).astype(np.float32)

    with open(data_info_file_path, 'w') as f:
        json.dump( #spectrogram_img_info_arr, f)
            {
                "max" : float(sample_max),
                "min" : float(sample_min),
                "sample_indices" : spectrogram_img_info_arr
            }, f)

#%%
def create_samples_from_audio(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size):
    _, _, spectrogram_img_arr, spectrogram_img_info_arr = create_spectrogram_slices(sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
    
    data_path = os.path.join(dataset_path, 'data')
    samples_file_path = os.path.join(data_path, f"{sample_id}.bin")

    data_info_path = os.path.join(dataset_path, 'data_info')
    data_info_file_path = os.path.join(data_info_path, f"{sample_id}.json")

    np.asarray(spectrogram_img_arr, dtype = np.float32).tofile(samples_file_path)

    with open(data_info_file_path, 'w') as f:
        json.dump(spectrogram_img_info_arr, f)

    img_max = np.max(spectrogram_img_arr).astype(np.float32)
    img_min = np.min(spectrogram_img_arr).astype(np.float32)
    return img_max, img_min

#%%
def create_samples_from_audio_samples(dataset_path, sample_ids, sample_rate, stft_window_size, stft_step_size,  frequency_range_size, lower_frequency_margin, img_dim, img_step_size, event_padding_size):
    for sample_id in sample_ids:
        create_samples_from_audio(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  frequency_range_size, lower_frequency_margin, img_dim, img_step_size, event_padding_size)

#%%
def equal_split(arr, n_chunks):
    chunk_size, m = divmod(len(arr), n_chunks)
    return [arr[index * chunk_size + min(index, m) : (index + 1) * chunk_size + min(index + 1, m)] for index in range(n_chunks)]

#%%
def pytorch_create_and_save_dateset(
        dataset_path, 
        sample_ids, 
        sample_rate, 
        stft_window_size, 
        stft_step_size, 
        frequency_range_size, 
        lower_frequency_margin, 
        sample_dim, 
        sampling_step_size, 
        sampling_padding_size, 
        num_workers = 1,
        device = 'cpu'
    ):
    
    # check if dataset directory already exists    
    if os.path.exists(dataset_path): 
        shutil.rmtree(dataset_path) # if so delete it
    # create dataset directory
    os.makedirs(dataset_path)
    # create dataset sub directories
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
            worker_inputs = [(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  frequency_range_size, lower_frequency_margin, sample_dim, sampling_step_size, sampling_padding_size, device) for sample_id in sample_ids]
            pool.starmap(pytorch_create_samples_from_audio, worker_inputs, chunksize=1) #setting chunksize to 1 to due to imbalanced sample processing time 
        
    else: # if only one worker is used, create the dataset sequentially on the main thread
        for sample_id in tqdm.tqdm(sample_ids):
            pytorch_create_samples_from_audio(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  frequency_range_size, lower_frequency_margin, sample_dim, sampling_step_size, sampling_padding_size, device)


    max_value = 0.0
    min_value = np.inf
    samples_count = 0

    for data_info_file in os.listdir(data_info_path): 
        with open(os.path.join(data_info_path, data_info_file), 'r') as f:
            data_info = json.load(f)
            max_value = max(max_value, data_info['max'])
            min_value = min(min_value, data_info['min'])
            samples_count += len(data_info['sample_indices'])

    end_time = datetime.now()

    with open(os.path.join(dataset_path, 'dataset_attributes.json'), 'w') as f:
        json.dump({
            "start_date" : start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date" : end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_taken_sec" : (end_time - start_time).total_seconds(),
            "samples_count" : samples_count,
            "sample_dim" : list(sample_dim),
            "max_value" : float(max_value),
            "min_value" : float(min_value),
            "parameters" : {
                "sample_rate" : sample_rate,
                "stft_window_size" : stft_window_size,
                "stft_step_size" : stft_step_size,
                "frequency_range_size" : frequency_range_size,
                "lower_frequency_margin" : lower_frequency_margin,

            }
        }, f, indent=2)


#%%
def create_and_save_dateset(dataset_path, sample_ids, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size, num_workers = 1):
    
    # check if dataset directory already exists    
    if os.path.exists(dataset_path): 
        shutil.rmtree(dataset_path) # if so delete it
    # create dataset directory
    os.makedirs(dataset_path)
    # create dataset sub directories
    data_path = os.path.join(dataset_path, 'data')
    data_info_path = os.path.join(dataset_path, 'data_info')

    os.makedirs(data_path)
    os.makedirs(data_info_path)

    print(f"Processing {len(sample_ids)} samples")
    num_workers = min(num_workers, len(sample_ids)) #set the number of workers to the number of samples if the specified worker number is greater than the number of samples
    print(f"Creating dataset with {num_workers} workers")

    start_time = datetime.now()
    
    max_value = 0.0
    min_value = np.inf

    if num_workers > 1: # if the workers are more than 1, create a processing pool
        with Pool(processes=num_workers) as pool:
            worker_inputs = [(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size) for sample_id in sample_ids]
            max_min_arr = pool.starmap(create_samples_from_audio, worker_inputs, chunksize=1) #setting chunksize to 1 to due to imbalanced sample processing time 

        max_min_arr = np.array(max_min_arr)
        max_value = np.max(max_min_arr[:, 0])
        min_value = np.min(max_min_arr[:, 1])
        
    else: # if only one worker is used, create the dataset sequentially on the main thread
        for sample_id in tqdm.tqdm(sample_ids):
            sample_max_value, sample_min_value = create_samples_from_audio(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
            max_value = np.max([max_value, sample_max_value])
            min_value = np.min([min_value, sample_min_value])
    
    end_time = datetime.now()

    with open(os.path.join(dataset_path, 'dataset_attributes.json'), 'w') as f:
        json.dump({
            "start_date" : start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date" : end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_taken_sec" : (end_time - start_time).total_seconds(),
            "img_dim" : list(img_dim),
            "max_value" : float(max_value),
            "min_value" : float(min_value),
        }, f, indent=2)

#%%
def create_and_return_dataset(sample_ids, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size, num_workers = 1):
    all_sample_events = []
    all_spectrogram_images = []
    for index, sample_id in tqdm.tqdm(enumerate(sample_ids), total=len(sample_ids)):
        # print(f"Sample: {index + 1:<5}/{len(sample_ids):<5} id: {sample_id}", end='\r')
        masked_spectrogram, audio_events_arr, spectrogram_img_arr, _ = create_spectrogram_slices(sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
        all_sample_events += audio_events_arr
        all_spectrogram_images += spectrogram_img_arr

    return all_spectrogram_images

if __name__ == "__main__":
    project_dir = os.getcwd()


    dataset_info = DatasetInfo()
    sample_ids = dataset_info.get_download_sample_ids(2473663, SampleRecordingType.Foreground)


    sample_rate  = 44100
    stft_window_size = 2048
    stft_step_size = 512
    max_frequency = 10000
    min_frequency = 2500

    img_dim = (32, 512) # (time, freq)
    img_step_size = 1  # (time)
    event_padding_size = 4

    num_workers = 24

    create_new = False

    dataset_path = os.path.join(project_dir, 'data', 'spectrogram_samples',f'test_samples_xd{img_dim[0]}_yd{img_dim[1]}_iss{img_step_size}')

    # if not os.path.exists(dataset_path) or create_new:
    #     print("Building dataset")
    #     #create_and_return_dataset(sample_ids[:12], sample_rate, stft_window_size, stft_step_size, max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
    #     create_and_save_dateset(dataset_path, sample_ids, sample_rate, stft_window_size, stft_step_size, max_frequency, min_frequency, img_dim, img_step_size, event_padding_size, num_workers=num_workers)
    #     print("Dataset built")
    #     print(f"Stored in location: {dataset_path}")
    # else:
    #     print(f"Dataset found. Loading dataset from {dataset_path}")
    # sample_id = sample_ids[0]
    
    #sample_ids = sample_ids[:10]

    # start_time = process_time()
    # print("current_method")
    # for sample_id in sample_ids:
    #     print(f"\n{sample_id}")
    #     audio_start_time = process_time()
    #     _, _, spectrogram_img_arr, _ = create_spectrogram_slices(sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
    #     print(f"Samples created: {len(spectrogram_img_arr)}")
    #     print(f"Audio Total time {process_time() - audio_start_time:.3f} sec")
    # print(f"Total time {process_time() - start_time:.3f} sec for {len(sample_ids)} samples")


    sample_rate  = 44100
    stft_window_size = 2048
    stft_step_size = 512

    # max_frequency = 10000
    # min_frequency = 2500

    sample_dim = (512, 32) # (freq, time) ###(32, 512) # (time, freq)
    sampling_step_size = 1  # (time)
    sampling_padding_size = sample_dim[1] // 2

    frequency_range_size = sample_dim[0]
    lower_frequency_margin = 100

    num_workers = 12

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(project_dir, 'data', 'spectrogram_samples',f'pt_samples_0d{sample_dim[0]}_1d{sample_dim[1]}_iss{sampling_step_size}')
    print("Building dataset")
        #create_and_return_dataset(sample_ids[:12], sample_rate, stft_window_size, stft_step_size, max_frequency, min_frequency, sample_dim, sampling_step_size, sampling_padding_size)
    pytorch_create_and_save_dateset(dataset_path, sample_ids, sample_rate, stft_window_size, stft_step_size, frequency_range_size, lower_frequency_margin, sample_dim, sampling_step_size, sampling_padding_size, num_workers=num_workers, device = device)
    print("Dataset built")
    print(f"Stored in location: {dataset_path}")

    # print("pytorch_method")
    # start_time = process_time()
    # for sample_id in sample_ids:
    #     print(f"\n{sample_id}")
    #     audio_start_time = process_time()
    #     _, _, spectrogram_img_arr, _ = torch_create_spectrogram_samples(sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
    #     print(f"Samples created: {len(spectrogram_img_arr)}")
    #     print(f"Audio Total time {process_time() - audio_start_time:.3f} sec")
    # print(f"Total time {process_time() - start_time:.3f} sec for {len(sample_ids)} samples")

    # import cProfile
    # cProfile.run(f'create_spectrogram_slices({sample_id}, {sample_rate}, {stft_window_size}, {stft_step_size}, {max_frequency}, {min_frequency}, {img_dim}, {img_step_size}, {event_padding_size})')
