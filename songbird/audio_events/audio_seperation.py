#%%
import pydub
import time
import numpy as np
import pandas as pd
import json
import os
import audio_processing as ap
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from dataclasses import dataclass

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        self.dense1 = nn.Linear(4410, 400) 
       # self.dense2 = nn.Linear(256, 128) 
        self.mean_dense = nn.Linear(400, 20)
        self.variance_dense = nn.Linear(400, 20)

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        #x =  F.relu(self.dense2(x))
        x_mean =  self.mean_dense(x)
        x_variance =  self.variance_dense(x)
        return  x_mean, x_variance

class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        self.dense1 = nn.Linear(20, 400)
        #self.dense2 = nn.Linear(128, 256) 
        self.dense3 = nn.Linear(400, 4410) 

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        #x =  F.relu(self.dense2(x))
        x =  F.sigmoid(self.dense3(x))
        return x  

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

def loss_function(x_hat, x, mean, variance):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + KLD


def mse_loss_function(x_hat, x, mean, variance):
    BCE = nn.functional.mse_loss(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + KLD


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



def trim_to_frequency_range(spectrogram, frequency_bins, max_frequency_index, min_frequency_index):
    # max_index = np.argmin(np.abs(frequency_bins - max_frequency_index))
    # print(max_frequency_index, min_frequency_index)
    # print(len(frequency_bins))
    # min_index = max_index - min_frequency_index
    # print(max_index, min_index)
    return crop_frequency_range(spectrogram, frequency_bins, max_frequency_index, min_frequency_index)
#%%
def create_spectrogram(sample, window_size, step_size, sample_rate, max_frequency, min_frequency):
    spectrogram, frequency_bins = ap.create_spectrogram(sample, window_size, step_size, sample_rate)
    # print(spectrogram.shape)

    #spectrogram, frequency_bins = trim_to_frequency_range(spectrogram, frequency_bins, 512, 0)
    # print(spectrogram.shape)
    cropped_spectrogram, frequency_bins = ap.trim_to_frequency_range(spectrogram, frequency_bins, max_frequency, min_frequency)
    print(spectrogram.shape, cropped_spectrogram.shape)
    cropped_spectrogram = cropped_spectrogram.numpy()
    #print(type(spectrogram))
    return cropped_spectrogram, frequency_bins

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


#%%
def create_spectrogram_slices(sample_id, audio_conversion_parameters, plot = False):
    
    if plot:
        ap.empty_or_create_dir(os.path.join(ap.plots_path, str(sample_id)))
    
    sample, _ = ap.load_audio_sample(sample_id, audio_conversion_parameters.sample_rate)

    sos = signal.butter(10, 3500, 'hp', fs=audio_conversion_parameters.sample_rate, output='sos')

    sample = signal.sosfilt(sos, sample)

    spectrogram, frequency_bins = create_spectrogram(sample, audio_conversion_parameters.window_size, audio_conversion_parameters.step_size, audio_conversion_parameters.sample_rate, audio_conversion_parameters.max_frequency, audio_conversion_parameters.min_frequency)

    mask = get_audio_mask(spectrogram, 1.2)
    
    audio_event_data_slices = get_audio_events(mask)

    audio_events_arr = []
    for time_slice, frequency_slice in audio_event_data_slices:
        audio_events_arr.append(spectrogram_audio_event(time_slice.start, time_slice.stop, frequency_slice.start, frequency_slice.stop))



#%%


#%%
sample_ids = [
    2243804495,
    2432423171,
    2243742980,
    2243784966,
    2243675399,
    2455252743,
    2432420110,
    2432417551,
    2243570965,
    2243667228,
    2243883806,
    2243740447,
    2243583779,
    2432421155,
    2243571495,
    2243587373,
    2243778866,
]
#%%
class AudioConversionParameter:
    def __init__(self) -> None:
        self.window_size = 2048
        self.step_size = 512
        self.sample_rate  = 44100
        self.max_frequency = 10000
        self.min_frequency = 2500

acp = AudioConversionParameter()
#%%
all_sample_events = []

for sample_id in sample_ids:
    print(sample_id)
    masked_spectrogram, audio_events_arr = create_masked_spectrogram(sample_id, acp)
    all_sample_events += audio_events_arr
    # print(masked_spectrogram.shape)    
    # plt.matshow(masked_spectrogram.T)
    # plt.gca().invert_yaxis()
    # plt.show()

    #ap.save_spectrogram_plot(masked_spectrogram.T, acp.sample_rate, acp.step_size, sample_id, title=f'{sample_id}', y_labels=rfft_bin_freq)
# %%
#%%
time_mean_width =  np.mean([event.end_time_index - event.start_time_index for event in all_sample_events])
time_std_with = np.std([event.end_time_index - event.start_time_index for event in all_sample_events])
time_mean_width, time_std_with

# %%
