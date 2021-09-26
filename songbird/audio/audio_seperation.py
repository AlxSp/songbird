#%%
import os, sys
import shutil

from songbird.dataset.dataset_info import DatasetInfo, SampleRecordingType
import songbird.audio.audio_processing as ap

import pydub
import time
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

# import istarmap
from multiprocessing import Pool

#%%
@dataclass 
class spectrogram_audio_event:
    start_time_index: int
    end_time_index: int
    min_frequency_index: int
    max_frequency_index: int

@dataclass
class time_series_audio_event:
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



def trim_to_frequency_range(spectrogram, frequency_bins, max_frequency, to_min_frequency_window_size):
    #get closest frequency in bin
    max_index = 512 + 100#np.argmin(np.abs(frequency_bins - max_frequency))
    min_index = 0 + 100
    #min_index = max_index - to_min_frequency_window_size 
    #min_index = np.argmin(np.abs(frequency_bins - min_freq))

    # print(f'Min frequency index: {min_index}')
    # print(f'Max frequency index: {max_index}')
    # print(frequency_bins)
    return spectrogram[:,min_index:max_index], frequency_bins[min_index:max_index]

#%%
def get_audio_mask(spectrogram, std_threshold):
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
    
    sample, _ = ap.load_audio_sample(sample_id, sample_rate)

    sos = signal.butter(10, 3500, 'hp', fs=sample_rate, output='sos')

    sample = signal.sosfilt(sos, sample)

    spectrogram, frequency_bins = ap.create_spectrogram(sample, stft_window_size, stft_step_size, sample_rate)

    cropped_spectrogram, frequency_bins = trim_to_frequency_range(spectrogram, frequency_bins, max_frequency, img_dim)
    # print(spectrogram.shape, cropped_spectrogram.shape)
    cropped_spectrogram = cropped_spectrogram.numpy()

    mask = get_audio_mask(cropped_spectrogram, 1.2)
    
    audio_event_data_slices = get_audio_events(mask)

    audio_events_arr = []
    for time_slice, frequency_slice in audio_event_data_slices:
        audio_events_arr.append(spectrogram_audio_event(time_slice.start, time_slice.stop, frequency_slice.start, frequency_slice.stop))

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

            #check if new event start is within the actual spectrogram
            if new_audio_event_start_index < 0: #if event index is negative, set it to zero and move the end index by the absolute value of the out bounds index
                out_of_range_index = abs(new_audio_event_start_index)
                new_audio_event_start_index += out_of_range_index
                new_audio_event_end_time_index += out_of_range_index
            #check if new event end is within the actual spectrogram            
            if new_audio_event_end_time_index > cropped_spectrogram.shape[0]:
                new_audio_event_end_time_index = cropped_spectrogram.shape[0]
                new_audio_event_start_index = new_audio_event_end_time_index - img_dim[0] if new_audio_event_end_time_index - img_dim[0] > 0 else 0
            
            new_event_length = new_audio_event_end_time_index - new_audio_event_start_index
            noise_data = cropped_spectrogram[np.where(np.invert(mask))]
            noise_mean = np.mean(noise_data)
            noise_std = np.std(noise_data)

            # print(f"Noise mean {noise_mean} noise std {noise_std}")
            #print(noise_data.shape)
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


    return cropped_spectrogram * mask , audio_events_arr, spectrogram_img_arr, spectrogram_img_info_arr

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
def create_samples_from_audio_samples(dataset_path, sample_ids, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size):
    for sample_id in sample_ids:
        create_samples_from_audio(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)

#%%
def equal_split(arr, n_chunks):
    chunk_size, m = divmod(len(arr), n_chunks)
    return [arr[index * chunk_size + min(index, m) : (index + 1) * chunk_size + min(index + 1, m)] for index in range(n_chunks)]

#%%
def create_and_save_dateset(dataset_path, sample_ids, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size, num_workers = 1):
    data_path = os.path.join(dataset_path, 'data')
    data_info_path = os.path.join(dataset_path, 'data_info')

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        #os.rmdir(dateset_path)

    os.makedirs(dataset_path)
    os.makedirs(data_path)
    os.makedirs(data_info_path)

    max_value = 0.0
    min_value = np.inf

    num_workers = min(num_workers, len(sample_ids))
    print(f"Creating dataset with {num_workers} workers")
    if num_workers > 1:
        #chunked_sample_ids = list(equal_split(sample_ids, num_workers))
        with Pool(processes=num_workers) as pool:
            worker_inputs = [(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size) for sample_id in sample_ids]
            max_min_arr = pool.starmap(create_samples_from_audio, worker_inputs)#p.starmap(create_samples_from_audio, worker_inputs)


        max_min_arr = np.array(max_min_arr)
        max_value = np.max(max_min_arr[:, 0])
        min_value = np.min(max_min_arr[:, 1])
        
    else: 
        for sample_id in tqdm.tqdm(sample_ids):
            sample_max_value, sample_min_value = create_samples_from_audio(dataset_path, sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
            max_value = np.max([max_value, sample_max_value])
            min_value = np.min([min_value, sample_min_value])

    with open(os.path.join(dataset_path, 'data_attributes.json'), 'w') as f:
        json.dump({
            "img_dim" : list(img_dim),
            "max_value" : float(max_value),
            "min_value" : float(min_value),
        }, f)

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

    dataset_path = os.path.join(project_dir, 'data', 'spectrogram_samples',f'samples_xd{img_dim[0]}_yd{img_dim[1]}_iss{img_step_size}')

    if not os.path.exists(dataset_path) or create_new:
        print("Building dataset")
        #create_and_return_dataset(sample_ids[:12], sample_rate, stft_window_size, stft_step_size, max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
        create_and_save_dateset(dataset_path, sample_ids, sample_rate, stft_window_size, stft_step_size, max_frequency, min_frequency, img_dim, img_step_size, event_padding_size, num_workers=num_workers)
        print("Dataset built")
        print(f"Stored in location: {dataset_path}")
    else:
        print(f"Dataset found. Loading dataset from {dataset_path}")
