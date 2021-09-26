#%%
import os, sys
import shutil

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)

from songbird.data.dataset_info import DatasetInfo, SampleRecordingType
import audio_processing as ap

import pydub
import time
import numpy as np
import pandas as pd
import tqdm
import json
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy as sp
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from skimage.feature import peak_local_max

import istarmap
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        #if the stride is not 1 or the input and output channels are not the same, transform residual value into the right shape
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.residual = nn.Identity()
        

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        # print(f"Conv 1 output shape: {x.shape}")
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # print(f"Conv 2 output shape: {x.shape}")
        x = self.batch_norm2(x)
        x = x + residual
        x = F.relu(x)

        return x

class TransposeResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,):
        super(TransposeResidualBlock2d, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        residual =  x
        x = self.tconv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.tconv2(x)
        x = self.batch_norm2(x)
        x = x + residual
        x = F.relu(x)

        return x

#%%
class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()

        self.res_block1 = ResidualBlock2d(in_channels=1, out_channels=16)
        self.res_block2 = ResidualBlock2d(in_channels=16, out_channels=32)
        self.res_block3 = ResidualBlock2d(in_channels=32, out_channels=64)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        # self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        #self.conv5 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        #self.dense1 = nn.Linear(4410, 400) 
        
        self.mean_dense = nn.Linear(16384, 256)
        self.variance_dense = nn.Linear(16384, 256)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.res_block1(x)#F.relu(self.conv1(x))
        # print(f"Res 1 output shape: {x.shape}")
        x = self.max_pool1(x)

        x = self.res_block2(x)#F.relu(self.conv2(x))
        # print(f"Res 2 output shape: {x.shape}")
        x = self.max_pool2(x)
        
        x = self.res_block3(x)#F.relu(self.conv3(x))
        # print(f"Res 3 output shape: {x.shape}")
        x = self.max_pool3(x)
        #x = F.relu(self.conv4(x))
        # print(f"Conv 4 output shape: {x.shape}")
        #x = F.relu(self.conv5(x))
        # print(f"Conv 5 output shape: {x.shape}")

        # x = self.pool(x)
        # print(f"Pool output shape: {x.shape}")
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        # print(f"Flattened output shape: {x.shape}")

        #x = F.adaptive_avg_pool2d(x, 3).reshape(batch_size, -1)
        #print(f"Flattened output shape: {x.shape}")
        # x = self.dense(x)

        x_mean =  self.mean_dense(x)
        x_variance =  self.variance_dense(x)
        
        return  x_mean, x_variance
#%%
class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        
        self.dense1 = nn.Linear(256, 4096)
        #self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):

        # print(f"Input shape: {x.shape}")
        x =  F.relu(self.dense1(x))
        # print(f"Dense output shape: {x.shape}")
        x = x.view(-1, 64, 2, 32) #x.view(-1, 128, 1, 16)
        # print(f"Reshape shape: {x.shape}")
        #x = F.relu(self.conv1(x))
        # print(f"Conv 1 output shape: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"Conv 2 output shape: {x.shape}")
        x = F.relu(self.conv3(x))
        # print(f"Conv 4 output shape: {x.shape}")
        x = F.relu(self.conv4(x))
        # print(f"Conv 4 output shape: {x.shape}")
        x =  F.sigmoid(self.conv5(x))
        # print(f"Conv 5 output shape: {x.shape}")
        #x = x.view(-1, 64, 1, 1)
        #x =  F.relu(self.dense2(x))
        return x  
#%%
class VariationalAutoDecoder(nn.Module):  
    def __init__(self, encoder, decoder):
        super(VariationalAutoDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x_mean, x_variance = self.encoder(x)

        if self.training:
            std = torch.exp(x_variance * 0.5)
            eps = torch.randn_like(std) 
            z = x_mean + (eps * std)
        else:
            z = x_mean

        x = self.decoder(z)
        return x, x_mean, x_variance

#%%
def loss_function(x_hat, x, mean, variance, beta = 1.0):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + beta * KLD


def mse_loss_function(x_hat, x, mean, variance, beta = 1.0):
    BCE = nn.functional.mse_loss(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + beta * KLD


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
def show_audio(x, y, label = None, fig_num = 1, plot_num = 4, sample_rate = None, report_dir = None):
    for fig_index in range(fig_num):
        fig, axs = plt.subplots(2, 4)
        in_audio = x.data.cpu()
        #axs.suptitle(label + ' - real test data / reconstructions', color='w', fontsize=16)
        for i in range(plot_num):
            axs[0, i].plot(in_audio[fig_index * plot_num + i])#plt.subplot(1, 4, i + 1)
            # axs[1, i].set_xlabel('Time (sec)')
            
            # axs[0, i].axis('off')
            #plt.imshow(in_audio[i+4+plot_index])
            #plt.axis('off')
        out_audio = y.data.cpu()
        #plt.figure(figsize=(18,6))
        for i in range(plot_num):
            #plt.subplot(1, 4, i + 1)
            axs[1, i].plot(out_audio[fig_index * plot_num + i])
            # axs[1, i].set_xlim(left = 0, right = len(out_audio[i+4+N]))
            # axs[1, i].set_xlabel('Time (sec)')
            # axs[1, i].axis('off')

        fig.canvas.set_window_title(f"{label} - real test data / reconstructions'")
        plt.savefig(os.path.join(report_dir, f'{label}_{fig_index}.png'))
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

# %%
def show_img_sample(img):
    plt.matshow(img.T)
    plt.gca().invert_yaxis()
    plt.show()

# %%
def show_x_vs_y_samples(x, y, sample_dim, tile=None, column_headers = [], row_headers = [], fig_num = 1, plot_num = 4, report_dir = None):
    for fig_index in range(fig_num):
        plot_num = min(len(x), plot_num)
        
        fig, axes = plt.subplots(nrows = 2, ncols = plot_num, squeeze=False, sharex=True, sharey=True)
        fig.suptitle(f"{tile} - Real Data (X) vs Reconstruction (X Hat)")
        in_pic = x.data.cpu().view(-1, *sample_dim)
        #axs.suptitle(label + ' - real test data / reconstructions', color='w', fontsize=16)
        #axs[0].set_title(f"{label} - x")
        for i in range(plot_num):
            axes[0, i].imshow(in_pic[fig_index * plot_num + i])#plt.subplot(1, plot_num, i + 1)
            axes[0, i].axis('off')

        out_pic = y.data.cpu().view(-1, *sample_dim)
        #plt.figure(figsize=(18,6))
        #axs[1].set_title(f"{label} - y")
        for i in range(plot_num):
            axes[1, i].imshow(out_pic[fig_index * plot_num + i])
            axes[1, i].axis('off')

        for ax, col in zip(axes[0,:], column_headers):
            ax.set_title(col, size=10)

        for ax, row in zip(axes[:,0], row_headers): #["X", "X Hat"]
            ax.annotate(row, (0, 0.5), xytext=(-25, 0), ha='right', va='center',
                size=15, rotation=90, xycoords='axes fraction',
                textcoords='offset points')

        # plt.tight_layout(pad = 0.5)
        fig.canvas.set_window_title(f"{tile} - real test data / reconstructions'")
        plt.savefig(os.path.join(report_dir, f'{tile}_{fig_index}.png'))
        writer.add_figure(f"{tile} - real test data / reconstructions", fig, None)

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
        self.img_dim = img_dim
        self.transform = transform

        with open(os.path.join(self.dataset_path, "data_attributes.json"), 'r') as f:
            self.dataset_attributes = json.load(f)
        # calculate the size of one sample in bytes; the sample's dimensions multiplied with each other, multiplied by the data type's byte size that was used to store the samples in the file
        self.sample_byte_size = np.prod(self.dataset_attributes["img_dim"]) * np.dtype(np.float32).itemsize

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
    # print(masked_spectrogram.shape)    
    #how_img_sample(masked_spectrogram)

    # np.save(samples_path, all_spectrogram_images)

    return all_spectrogram_images

#%%
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
#samples_path = os.path.join(project_dir, 'data', 'spectrogram_samples', f'samples_xd{img_dim[0]}_yd{img_dim[1]}_iss{img_step_size}.npy')

# %%
all_spectrogram_images = []
if not os.path.exists(dataset_path) or create_new:
    print("Building dataset")
    #create_and_return_dataset(sample_ids[:12], sample_rate, stft_window_size, stft_step_size, max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
    create_and_save_dateset(dataset_path, sample_ids, sample_rate, stft_window_size, stft_step_size, max_frequency, min_frequency, img_dim, img_step_size, event_padding_size, num_workers=num_workers)
    print("Dataset built")
    print(f"Stored in location: {dataset_path}")
else:
    print(f"Dataset found. Loading dataset from {dataset_path}")
    #all_spectrogram_images = np.load(samples_path)
    #ap.save_spectrogram_plot(masked_spectrogram.T, acp.sample_rate, acp.step_size, sample_id, title=f'{sample_id}', y_labels=rfft_bin_freq)
# %%
#all_spectrogram_images = min_max_normalize(all_spectrogram_images)

#%%
#print(f"Spectrogram dataset shape: {all_spectrogram_images.shape}\n")
#%%
# show_img_sample(all_spectrogram_images[0])
# show_img_sample(all_spectrogram_images[32])
# show_img_sample(all_spectrogram_images[64])

def print_process(epoch, train_loss, test_loss):
    print(f"epoch: {epoch:5} | train loss: {train_loss / len(train_loader.dataset):16.6f} | test loss: {test_loss / len(test_loader.dataset):16.6f}", end = "\r")


def get_run_dir(run_name):
    runs_dir = os.path.join(project_dir, 'runs')
    run_index = 0
    run_dir = os.path.join(runs_dir, f'{run_name}_{run_index}')
    while os.path.exists(run_dir):
        run_index += 1
        run_dir = os.path.join(runs_dir, f'{run_name}_{run_index}')

    
    return run_dir

#%%
model_name = 'vae_spectrogram'
run_dir = get_run_dir(model_name)

writer = SummaryWriter(log_dir=run_dir)
print(f"Saving run data to dir: {run_dir}")
#%%
report_dir = os.path.join(ap.project_base_dir, "reports", "vae", "songbird_model")

learning_rate = 1e-4
epochs = 200 
batch_size = 1024 // 8

val_percentage = 0.05
random_seed = 42

writer.add_hparams(
    {   "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "val_percentage": val_percentage,
        "random_seed": random_seed,
        "Optimizer": "AdamW",
    },
    dict({
        "data_processing/sample_rate" : sample_rate,
        "data_processing/stft_window_size" : stft_window_size,
        "data_processing/stft_step_size" : stft_step_size,
        "data_processing/max_frequency" : max_frequency,
        "data_processing/min_frequency" : min_frequency,

        "data_processing/img_step_size" : img_step_size,  # (time)
        "data_processing/event_padding_size" : event_padding_size
    }, **{f'data_processing/img_dim_{i}': v for i, v in enumerate(img_dim)})
    
)

plot_params = {
    "report_dir" : report_dir,
    "plot_num" : 4,
    "sample_rate": sample_rate
}


# %%
spectrogram_dataset = SpectrogramFileDataset(dataset_path, transform=ToTensor())
print(f"{'#'*3} {'Dataset info' + ' ':{'#'}<{24}}")
print(f"Total dataset length: {len(spectrogram_dataset)}")

#%%
train_size = int(len(spectrogram_dataset) * (1 - val_percentage))
test_size = len(spectrogram_dataset) - train_size #int(len(spectrogram_dataset) * val_percentage)
train_set, val_set = torch.utils.data.random_split(spectrogram_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))

# %%
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=12)
#%%
print(f"Train length: {train_size} Test length: {test_size}")
print(f"Train batch num: {len(train_loader)} Test batch num: {len(test_loader)}")
# %%
encoder = VariationalEncoder()
decoder = VariationalDecoder()
model = VariationalAutoDecoder(encoder, decoder)

writer.add_graph(model, next(iter(train_loader))[0]) # add model to tensorboard and
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'#'*3} {'Training info' + ' ':{'#'}<{24}}")
print(f"Using {device} device for training")
model.to(device)
model.train()

#codes = dict(mu = list(), variance = list(), y=list())
#%%
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#%%
print(f"{'#'*3} {'Training' + ' ':{'#'}<{24}}")
for epoch in range(0, epochs):
    train_loss = 0
    test_loss = 0
    
    samples_count = 0
    if epoch > 0:
        model.train()
        for batch, _, _ in train_loader:
            # print(f"x: {x}")
            batch = batch.to(device)
            x_hat, mu, logvar = model(batch)
            loss = loss_function(x_hat, batch, mu, logvar)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            samples_count += len(batch)
            print(f"epoch: {epoch:5} | train loss: {train_loss / samples_count:16.6f} | test loss: {test_loss / len(test_loader.dataset):16.6f}", end = "\r")

        writer.add_scalar("Loss/train", train_loss / len(train_loader.dataset), epoch)

    # print(f"tain set length: {len(train_loader.dataset)} samples_count: {samples_count}")

    val_x = None
    val_x_hat = None
    file_paths = None
    sample_indices = None

    means, variance, labels = list(), list(), list()

    with torch.no_grad():
        model.eval()
        for batch, file_paths, sample_indices in test_loader:
            batch = batch.to(device)

            x_hat, mu, logvar = model(batch)

            test_loss += loss_function(x_hat, batch, mu, logvar).item()

            means.append(mu.detach())
            variance.append(logvar.detach())
            labels.append(batch.detach())

            #if len(x) > plot_params["plot_num"]:
            val_x = batch
            val_x_hat = x_hat

        print(f"epoch: {epoch:5} | train loss: {train_loss / len(train_loader.dataset):16.6f} | test loss: {test_loss / len(test_loader.dataset):16.6f}", end = "\r")
        
        writer.add_scalar("Loss/test", test_loss / len(test_loader.dataset), epoch)

        show_x_vs_y_samples(
            val_x, 
            val_x_hat, 
            sample_dim = img_dim, 
            tile = f'Epoch_{epoch}', 
            column_headers = [f"file: {os.path.splitext(os.path.basename(file_path))[0]}\nsample: {sample_index}" for file_path, sample_index in zip(file_paths, sample_indices)], 
            row_headers = ["X", "X Hat"], 
            fig_num = 1, 
            plot_num = plot_params["plot_num"], 
            report_dir = plot_params["report_dir"]
        )

    print()
#%%
# time_mean_width =  np.mean([event.end_time_index - event.start_time_index for event in all_sample_events])
# time_std_with = np.std([event.end_time_index - event.start_time_index for event in all_sample_events])
# time_mean_width, time_std_with