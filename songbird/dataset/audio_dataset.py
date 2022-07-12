import songbird.audio.audio_processing as ap

import os, sys
import json
import numpy as np
from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import Dataset



class ToTensor(object):
    def __call__(self, sample):
        sample = torch.reshape(torch.from_numpy(sample).float(), (1, sample.shape[0]))
        return sample

class PitchShift(object):
    def __init__(self, sample_rate, max_step = 15):
        self.sample_rate = sample_rate
        self.max_step = max_step
    def __call__(self, sample):
        return torchaudio.functional.pitch_shift(sample, self.sample_rate, torch.randint(-self.max_step, self.max_step, (1,))[0])

def get_audio_event(sample, sample_events, event_index, buffer = 0):
    try:
        return sample[ sample_events[event_index]['start'] - buffer : sample_events[event_index]['end'] + buffer ]
    except IndexError:
        print(f"Event index '{event_index}' exceeds number of events '{len(sample_events)}' in sample ")

def get_sample_audio_event(sample_arr, sample_event_arr, sample_index, event_index, buffer = 0):
    try:
        return get_audio_event(sample_arr[sample_index], sample_event_arr[sample_index], event_index, buffer)
    except IndexError:
        print(f"Sample index '{sample_index}' exceeds number of samples '{len(sample_arr)}'")

class AudioEventsDataset(Dataset):
    def __init__(self, audio_ids, audio_dir, audio_event_dir, sample_rate, sample_dim, step_size):
        self.sample_rate = sample_rate

        self.samples = []

        for audio_id in audio_ids:
            audio_arr = ap.load_audio_sample(audio_id, sample_rate, audio_dir)[0]
            audio_events = ap.load_audio_events(audio_id)
            for event_index in range(len(audio_events)):
                audio_of_event = get_audio_event(audio_arr, audio_events, event_index)

                for index in range(0, len(audio_of_event), step_size):
                    end_index = index + sample_dim
                    if end_index < index + sample_dim: #if the reduced train data slice is smaller index then the model input 
                        continue
                    self.samples.append(audio_of_event[index:end_index])

        self.samples = np.asarray(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class AudioFileDataset(Dataset):
    def __init__(self, 
                 dataset_path,
                 transform = None) -> None:

        self.dataset_path = dataset_path
        self.data_path = os.path.join(self.dataset_path, "data")
        self.data_info_path = os.path.join(self.dataset_path, "data_info")
        self.sample_dim = None
        self.transform = transform

        with open(os.path.join(self.dataset_path, "dataset_attributes.json"), 'r') as f:
            self.dataset_attributes = json.load(f)

        self.sample_dim = self.dataset_attributes["build_parameters"]["sample_dim"]
        self.data_type_size =  np.dtype(np.float32).itemsize
        self.sample_size = np.prod(self.sample_dim)

        #get all files in the data_info directory
        samples_info_paths = os.listdir(self.data_info_path)
        self.samples_file_paths = [os.path.join(self.data_path, samples_info_path.replace(".json", ".bin")) for samples_info_path in samples_info_paths]

        samples_to_file_indices = [] # array to store each file's starting sample index 6
        self.samples_start_offsets = [] # array to store each file's starting sample offset
        file_start_sample_index = 0  # used keep track of the first sample index of each file
        for samples_info_path in tqdm(samples_info_paths): # for each file in the data_info directory
            with open(os.path.join(self.data_info_path, samples_info_path), 'r') as f: # open the file
                samples_to_file_indices.append(file_start_sample_index) # add the file's starting sample index to the array
                sample_offsets = json.load(f)["sample_offsets"] # get the file's sample offsets
                samples_in_file = len(sample_offsets)   # get the number of samples in the file
                sample_start_offsets = np.asarray([sample_offset["start_index"] * self.data_type_size for sample_offset in sample_offsets], dtype = np.uint32)
                self.samples_start_offsets.append(sample_start_offsets) # add the file's starting sample offsets to the array
                file_start_sample_index += samples_in_file  # add the number of samples in the file to the file's starting sample index
                
        self.samples_to_file_indices = np.array(samples_to_file_indices)
        self.total_samples = file_start_sample_index # the total number of samples in the dataset is the sum of the number of samples in each file


    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        file_index = np.abs(self.samples_to_file_indices - index).argmin() #find the closest file by checking what file's starting sample index is closest to the index
        file_index = file_index if self.samples_to_file_indices[file_index] - index <= 0 else file_index - 1 #if the file's starting sample index is larger than the index, choose the preceding file
        file_name = self.samples_file_paths[file_index] # get the file's name in which the sample is stored
        sample_index = index - self.samples_to_file_indices[file_index] #find the actual sample index in the file by subtracting the file's starting sample index from the index. This will be used as the offset in the file
        sample_offset = self.samples_start_offsets[file_index][sample_index] # get the sample's offset in the file 
        # print(f"index: {index} nearest file index: {file_index} file index: {self.samples_to_file_indices[file_index]} sample index: {sample_index}")

        #read the binary file with the sample index as the offset times the size of a sample in bytes. Read in the amount of floats that make up one sample. Reshape the sample into the it's original shape
        sample = np.fromfile(self.samples_file_paths[file_index], offset = sample_offset, dtype = np.float32, count = self.sample_size).reshape(self.sample_dim) # TODO: apparently np.fromfile is slow, look into alternatives
        #sample = (sample - self.dataset_attributes["min_value"]) / (self.dataset_attributes["max_value"] - self.dataset_attributes["min_value"]) #normalize the sample

        if self.transform:
            sample = self.transform(sample)


        return sample, file_name, sample_index
