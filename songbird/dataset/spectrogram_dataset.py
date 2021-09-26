import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

#%%
class ToTensor(object):

    def __call__(self, sample):
        sample = torch.reshape(torch.from_numpy(sample).float(), (1, sample.shape[0], sample.shape[1]))
        return sample

#%%
class SpectrogramFileDataset(Dataset):
    def __init__(self, 
                 dataset_path,
                 transform = None) -> None:

        self.dataset_path = dataset_path    
        self.data_path = os.path.join(self.dataset_path, "data")
        self.data_info_path = os.path.join(self.dataset_path, "data_info")
        self.img_dim = None
        self.transform = transform

        with open(os.path.join(self.dataset_path, "data_attributes.json"), 'r') as f:
            self.dataset_attributes = json.load(f)
        # calculate the size of one sample in bytes; the sample's dimensions multiplied with each other, multiplied by the data type's byte size that was used to store the samples in the file
        self.img_dim = self.dataset_attributes["img_dim"]
        self.sample_byte_size = np.prod(self.img_dim) * np.dtype(np.float32).itemsize

        #get all files in the data_info directory
        samples_info_paths = os.listdir(self.data_info_path)
        #conver
        self.samples_file_paths = [os.path.join(self.data_path, samples_info_path.replace(".json", ".bin")) for samples_info_path in samples_info_paths]

        samples_to_file_indices = [] # array to store each file's starting sample index 
        samples_index = 0
        for samples_info_path in samples_info_paths:
            with open(os.path.join(self.data_info_path, samples_info_path), 'r') as f:
                samples_to_file_indices.append(samples_index)
                samples_in_file = len(json.load(f))
                samples_index += samples_in_file
                
        self.samples_to_file_indices = np.array(samples_to_file_indices)
        self.total_samples = samples_index


    def __len__(self):
        return self.total_samples


    def __getitem__(self, index):
        file_index = np.abs(self.samples_to_file_indices - index).argmin() #find the closest file by checking what file's starting sample index is closest to the index
        file_index = file_index if self.samples_to_file_indices[file_index] - index <= 0 else file_index - 1 #if the file's starting sample index is larger than the index, choose the preceding file
        file_name = self.samples_file_paths[file_index]
        sample_index = index - self.samples_to_file_indices[file_index] #find the actual sample index in the file by subtracting the file's starting sample index from the index. This will be used as the offset in the file
        # print(f"index: {index} nearest file index: {file_index} file index: {self.samples_to_file_indices[file_index]} sample index: {sample_index}")

        #read the binary file with the sample index as the offset times the size of a sample in bytes. Read in the amount of floats that make up one sample. Reshape the sample into the it's original shape
        sample = np.fromfile(self.samples_file_paths[file_index], offset = sample_index * self.sample_byte_size, dtype = np.float32, count = self.img_dim[0] * self.img_dim[1]).reshape(self.img_dim)
        sample = (sample - self.dataset_attributes["min_value"]) / (self.dataset_attributes["max_value"] - self.dataset_attributes["min_value"]) #normalize the sample

        if self.transform:
            sample = self.transform(sample)


        return sample, file_name, sample_index

# %%
class SpectrogramDataset(Dataset):
    def __init__(self, 
                 spectrogram_images,
                 transform = None) -> None:
        self.spectrogram_images = spectrogram_images
        self.transform = transform

    def __len__(self):
        return len(self.spectrogram_images)

    def __getitem__(self, idx):
        sample = self.spectrogram_images[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample