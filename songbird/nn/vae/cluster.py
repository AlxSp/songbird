#%%
from songbird.dataset.spectrogram_dataset import SpectrogramFileDataset, ToTensor
from songbird.dataset.dataset_info import DatasetInfo
from songbird.nn.vae.models.conv_vae import VariationalEncoder, VariationalDecoder, VariationalAutoEncoder
#%%
import os
import numpy as np
from sklearn.manifold import TSNE

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

#%%
random_seed = 42

samples_to_project = 10000

#%%
sample_rate  = 44100
stft_window_size = 2048
stft_step_size = 512
max_frequency = 10000
min_frequency = 2500

img_dim = (512, 32) # (freq, time)  # (time, freq)
img_step_size = 1  # (time)
event_padding_size = 4

num_workers = 12

batch_size = 256


model_name = 'conv_vae'
check_point_file = "checkpoint_e220.pt"

project_dir = os.getcwd()
model_dir = os.path.join('/home/alex/Dev/songbird/', 'models', model_name)
dataset_path = '/home/alex/Dev/songbird/data/spectrogram_samples/test_pt_samples_0d512_1d32_iss1/test'

#dataset_path = os.path.join(project_dir, 'data', 'spectrogram_samples',f'pt_samples_0d{img_dim[0]}_1d{img_dim[1]}_iss{img_step_size}')

#%%
np.random.seed(random_seed)

#%%
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
#%%
model.load_state_dict(torch.load(os.path.join(model_dir, check_point_file))['model_state_dict'])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'#'*3} {'Inference info' + ' ':{'#'}<{24}}")
print(f"Using {device} device for clustering")
model.to(device)
model.eval()
#%%
mean_arr = []
variance_arr = []
file_paths_arr = []
#%%
with torch.no_grad():
    for i, (x, file_names, sample_indices) in enumerate(data_loader):
        x = x.to(device)
        mean, variance = model.encoder(x)
        mean_arr.append(variance.cpu().numpy())
        file_paths_arr += file_names
# %%
mean_arr = np.concatenate(mean_arr)
#%%
to_project_indices = np.random.choice(np.arange(len(mean_arr)), size=samples_to_project, replace=False) if len(mean_arr) > samples_to_project else np.arange(len(mean_arr))
means_to_project = mean_arr[to_project_indices]

embedded_means = TSNE(n_components=2, init='random', random_state=random_seed).fit_transform(means_to_project)

#%%
dataset_info = DatasetInfo()
#%%
file_forground_bird_arr = []
file_background_bird_arr = []
file_id_arr = []
for file_path in file_paths_arr:
    file_id = os.path.basename(file_path).split('.')[0]
    meta_data = dataset_info.get_file_metadata(file_id)
    file_id_arr.append(file_id)
    file_forground_bird_arr.append(meta_data['forefront_bird_key'])
    file_background_bird_arr.append(meta_data['background_birds_keys'])

# %%
def plot_embedded_means(embedded_means, class_arr, title, cmap_color = 'rainbow'):
    cm = plt.cm.get_cmap(cmap_color)
    class_set = set(class_arr)
    class_colors = {class_key: cm(i) for class_key, i  in zip(class_set, np.linspace(0, 1, len(class_set)))}
    means_color_arr = [class_colors[class_key] for class_key in class_arr]


    fig, ax = plt.subplots(figsize=(10, 10))
    for class_key in class_set:
        class_indices = np.argwhere(np.asarray(class_arr) == class_key).flatten()
        class_embedded_means = embedded_means[class_indices]
        ax.scatter(class_embedded_means[:, 0], class_embedded_means[:, 1], c=[class_colors[class_key]], s=1, label = class_key)
    
    if len(class_set) > 1 and len(class_set) < 10:
        ax.legend()

    plt.title(title)
    plt.show()

#%%
foground_species_to_project = [file_forground_bird_arr[index] for index in to_project_indices]
plot_embedded_means(embedded_means, foground_species_to_project, 'Embedded means of the latent space with species')
# %%
background_species_to_project = [file_background_bird_arr[index] for index in to_project_indices]
amount_of_background_birds_in_samples = [len(set(background_birds_keys)) for background_birds_keys in background_species_to_project]
plot_embedded_means(embedded_means, amount_of_background_birds_in_samples, 'Embedded means of the latent space with amount of background species')

# %%
file_ids_to_project = [file_id_arr[index] for index in to_project_indices]

print(f"Samples come from {len(set(file_ids_to_project))} different files")
plot_embedded_means(embedded_means, file_ids_to_project, 'Embedded means of the latent space with file id', cmap_color = 'prism')
# %%
