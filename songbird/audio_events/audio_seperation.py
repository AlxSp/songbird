#%%
import os, sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)

import pydub
import time
import numpy as np
import pandas as pd
import json
from songbird.data.dataset_info import DatasetInfo, SampleRecordingType
import audio_processing as ap
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from dataclasses import dataclass



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

#%%
class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        #self.dense1 = nn.Linear(4410, 400) 
        
        self.mean_dense = nn.Linear(2048, 128)
        self.variance_dense = nn.Linear(2048, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(f"Conv 1 output shape: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"Conv 2 output shape: {x.shape}")
        x = F.relu(self.conv3(x))
        # print(f"Conv 4 output shape: {x.shape}")
        x = F.relu(self.conv4(x))
        # print(f"Conv 4 output shape: {x.shape}")
        x = F.relu(self.conv5(x))
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
        
        self.dense1 = nn.Linear(64, 2048)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):

        # print(f"Input shape: {x.shape}")
        x =  F.relu(self.dense1(x))
        # print(f"Dense output shape: {x.shape}")
        x = x.view(-1, 128, 1, 16)
        # print(f"Reshape shape: {x.shape}")
        x = F.relu(self.conv1(x))
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

    return cropped_spectrogram * mask , audio_events_arr, spectrogram_img_arr

# %%
def show_img_sample(img):
    plt.matshow(img.T)
    plt.gca().invert_yaxis()
    plt.show()

# %%
def show_x_vs_y_samples(x, y, sample_dim, label=None, fig_num = 1, plot_num = 4, report_dir = None):
    for fig_index in range(fig_num):
        plot_num = min(len(x), plot_num)
        
        fig, axes = plt.subplots(nrows = 2, ncols = plot_num, squeeze=False, sharex=True, sharey=True)
        fig.suptitle(f"{label} - Real Data (X) vs Reconstruction (X Hat)")
        in_pic = x.data.cpu().view(-1, *sample_dim)
        #axs.suptitle(label + ' - real test data / reconstructions', color='w', fontsize=16)
        #axs[0].set_title(f"{label} - x")
        for i in range(plot_num):
            axes[0, i].imshow(in_pic[fig_index * plot_num + i])#plt.subplot(1, plot_num, i + 1)
            axes[0, i].axis('off')
            #plt.imshow(in_pic[i+plot_num+plot_index])
            #plt.axis('off')

        out_pic = y.data.cpu().view(-1, *sample_dim)
        #plt.figure(figsize=(18,6))
        #axs[1].set_title(f"{label} - y")
        for i in range(plot_num):
            #plt.subplot(1, plot_num, i + 1)
            axes[1, i].imshow(out_pic[fig_index * plot_num + i])
            axes[1, i].axis('off')

        for ax, row in zip(axes[:,0], ["X", "X Hat"]):
            ax.annotate(row, (0, 0.5), xytext=(-25, 0), ha='right', va='center',
                size=15, rotation=90, xycoords='axes fraction',
                textcoords='offset points')

        fig.canvas.set_window_title(f"{label} - real test data / reconstructions'")
        plt.savefig(os.path.join(report_dir, f'{label}_{fig_index}.png'))

#%%
class ToTensor(object):

    def __call__(self, sample):
        sample = torch.reshape(torch.from_numpy(sample).float(), (1, sample.shape[0], sample.shape[1]))
        return sample

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
# sample_ids = [
#     2243804495,
#     2432423171,
#     2243742980,
#     2243784966,
#     2243675399,
#     2455252743,
#     2432420110,
#     2432417551,
#     2243570965,
#     2243667228,
#     2243883806,
#     2243740447,
#     2243583779,
#     2432421155,
#     2243571495,
#     2243587373,
#     2243778866,
# ]
#%%
dataset_info = DatasetInfo()
sample_ids = dataset_info.get_download_sample_ids(2473663, SampleRecordingType.Foreground)


sample_rate  = 44100
stft_window_size = 2048
stft_step_size = 512
max_frequency = 10000
min_frequency = 2500

img_dim = (32, 512) # (time, freq)
img_step_size = 4  # (time)
event_padding_size = 8


samples_path = os.path.join(project_dir, 'data', 'spectrogram_samples', f'samples_xd{img_dim[0]}_yd{img_dim[1]}_iss{img_step_size}.npy')



# %%
all_spectrogram_images = []
if not os.path.exists(samples_path):
    print("Dataset not found. Building dataset")
    all_sample_events = []
    all_spectrogram_images = []
    for index, sample_id in enumerate(sample_ids):
        print(f"Sample: {index + 1:<5}/{len(sample_ids):<5} id: {sample_id}", end='\r')
        masked_spectrogram, audio_events_arr, spectrogram_img_arr = create_spectrogram_slices(sample_id, sample_rate, stft_window_size, stft_step_size,  max_frequency, min_frequency, img_dim, img_step_size, event_padding_size)
        all_sample_events += audio_events_arr
        all_spectrogram_images += spectrogram_img_arr
    # print(masked_spectrogram.shape)    
    #how_img_sample(masked_spectrogram)

    np.save(samples_path, all_spectrogram_images)

else:
    print(f"Dataset found. Loading dataset from {samples_path}")
    all_spectrogram_images = np.load(samples_path)

    #ap.save_spectrogram_plot(masked_spectrogram.T, acp.sample_rate, acp.step_size, sample_id, title=f'{sample_id}', y_labels=rfft_bin_freq)
# %%

all_spectrogram_images = min_max_normalize(all_spectrogram_images)

#%%
print(f"Spectrogram dataset shape: {all_spectrogram_images.shape}\n")
#%%
# show_img_sample(all_spectrogram_images[0])
# show_img_sample(all_spectrogram_images[32])
# show_img_sample(all_spectrogram_images[64])

#%%
writer = SummaryWriter(log_dir=os.path.join(project_dir, 'runs', 'vae'))

#%%
report_dir = os.path.join(ap.project_base_dir, "reports", "vae", "songbird_model")

learning_rate = 1e-4
epochs = 100 
batch_size = 128

val_percentage = 0.05
random_seed = 42

plot_params = {
    "report_dir" : report_dir,
    "plot_num" : 4,
    "sample_rate": sample_rate
}


# %%
spectrogram_dataset = SpectrogramDataset(all_spectrogram_images, transform=ToTensor())

print(f"Total dataset length: {len(spectrogram_dataset)}")

#%%
train_size = int(len(spectrogram_dataset) * (1 - val_percentage))
test_size = len(spectrogram_dataset) - train_size #int(len(spectrogram_dataset) * val_percentage)
train_set, val_set = torch.utils.data.random_split(spectrogram_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))

# %%
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
#%%
print(f"Train length: {train_size} Test length: {test_size}")
print(f"Train batch num: {len(train_loader)} Test batch num: {len(test_loader)}")
# %%
encoder = VariationalEncoder()
decoder = VariationalDecoder()
model = VariationalAutoDecoder(encoder, decoder)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device for training")
model.to(device)
model.train()

#codes = dict(mu = list(), variance = list(), y=list())
#%%
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
for epoch in range(0, epochs):
    if epoch > 0:
        model.train()
        train_loss = 0
        for x in train_loader:
            # print(f"x: {x}")
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss = loss_function(x_hat, x, mu, logvar)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
        print(f"epoch: {epoch:4} | train loss: {train_loss / len(train_loader.dataset):10.6f}", end = "\r")

    val_x = None
    val_x_hat = None
    means, variance, labels = list(), list(), list()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for x in test_loader:
            x = x.to(device)

            x_hat, mu, logvar = model(x)

            test_loss += loss_function(x_hat, x, mu, logvar).item()

            means.append(mu.detach())
            variance.append(logvar.detach())
            labels.append(x.detach())

            #if len(x) > plot_params["plot_num"]:
            val_x = x
            val_x_hat = x_hat
        print()
        writer.add_scalar("Loss/test", test_loss / len(test_loader), epoch)
        print(f"epoch: {epoch:4} | test loss: {test_loss / len(test_loader.dataset):10.6f}")

    
    
    show_x_vs_y_samples(val_x, val_x_hat, img_dim, f'Epoch_{epoch}', 1, plot_params["plot_num"], plot_params["report_dir"])
#%%
# time_mean_width =  np.mean([event.end_time_index - event.start_time_index for event in all_sample_events])
# time_std_with = np.std([event.end_time_index - event.start_time_index for event in all_sample_events])
# time_mean_width, time_std_with