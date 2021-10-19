from songbird.dataset.spectrogram_dataset import SpectrogramFileDataset, ToTensor

from songbird.nn.vae.models.conv_vae import VariationalEncoder, VariationalDecoder, VariationalAutoEncoder

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader


img_dim = (512, 32) # (freq, time)  # (time, freq)
img_step_size = 1  # (time)
event_padding_size = 4

batch_size = 256


model_name = 'conv_vae'
check_point_file = "model_80.pt"

project_dir = os.getcwd()
model_dir = os.path.join(project_dir, 'models', model_name)
dataset_path = os.path.join(project_dir, 'data', 'spectrogram_samples',f'pt_samples_0d{img_dim[0]}_1d{img_dim[1]}_iss{img_step_size}')


spectrogram_dataset = SpectrogramFileDataset(dataset_path, transform=ToTensor())
print(f"{'#'*3} {'Dataset info' + ' ':{'#'}<{24}}")
print(f"Total dataset length: {len(spectrogram_dataset)}")

data_loader = DataLoader(spectrogram_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
# %%
encoder = VariationalEncoder()
decoder = VariationalDecoder()
model = VariationalAutoEncoder(encoder, decoder)


# checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')] #parse checkpoint files
# checkpoint_epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files] #get the epoch number from the checkpoint file name
# newest_checkpoint_file = checkpoint_files[np.argmax(checkpoint_epochs)] #get the newest checkpoint file

# checkpoint = torch.load(os.path.join(model_dir, newest_checkpoint_file))
# model.load_state_dict(checkpoint['model_state_dict'])

model.load_state_dict(torch.load(os.path.join(model_dir, check_point_file)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'#'*3} {'Training info' + ' ':{'#'}<{24}}")
print(f"Using {device} device for clustering")
model.to(device)
model.eval()

with torch.no_grad():
    for i, (x, _) in enumerate(data_loader):
        x = x.to(device)
