#%%
from songbird.dataset.dataset_info import DatasetInfo, SampleRecordingType
from songbird.dataset.spectrogram_dataset import SpectrogramFileDataset, ToTensor
from songbird.nn.vae.loss import loss_function
#from songbird.nn.vae.models.res_vae import VariationalEncoder, VariationalDecoder, VariationalAutoEncoder
from songbird.nn.vae.models.conv_vae import VariationalEncoder, VariationalDecoder, VariationalAutoEncoder

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

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
            axes[0, i].imshow(in_pic[fig_index * plot_num + i], origin = 'lower')#plt.subplot(1, plot_num, i + 1)
            axes[0, i].axis('off')

        out_pic = y.data.cpu().view(-1, *sample_dim)
        #plt.figure(figsize=(18,6))
        #axs[1].set_title(f"{label} - y")
        for i in range(plot_num):
            axes[1, i].imshow(out_pic[fig_index * plot_num + i], origin = 'lower')
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


def get_run_dir(model_runs_dir, prefix = 'run'):
    run_index = 0
    run_dir = os.path.join(model_runs_dir, f'{prefix}_{run_index}')
    while os.path.exists(run_dir):
        run_index += 1
        run_dir = os.path.join(model_runs_dir, f'{prefix}_{run_index}')


    return run_dir
#%%
dataset_info = DatasetInfo()
sample_ids = dataset_info.get_download_sample_ids(2473663, SampleRecordingType.Foreground)

project_dir = os.getcwd()

sample_rate  = 44100
stft_window_size = 2048
stft_step_size = 512
max_frequency = 10000
min_frequency = 2500

img_dim = (512, 32) # (freq, time)  # (time, freq)
img_step_size = 1  # (time)
event_padding_size = 4

num_workers = 12

continue_training = False

# dataset_path = os.path.join(project_dir, 'data', 'spectrogram_samples',f'pt_samples_0d{img_dim[0]}_1d{img_dim[1]}_iss{img_step_size}')

trainset_path = os.path.join(project_dir, 'data', 'spectrogram_samples',f'test_pt_samples_0d{img_dim[0]}_1d{img_dim[1]}_iss{img_step_size}', 'train')
testset_path = os.path.join(project_dir, 'data', 'spectrogram_samples',f'test_pt_samples_0d{img_dim[0]}_1d{img_dim[1]}_iss{img_step_size}', 'test')


#%%
model_name = 'conv_vae'

model_dir = os.path.join(project_dir, 'models', model_name)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

report_dir = os.path.join(project_dir, "reports", "vae", "songbird_model")

model_runs_dir = os.path.join(project_dir, 'runs', model_name)
run_dir = get_run_dir(model_runs_dir)

writer = SummaryWriter(log_dir=run_dir)
print(f"Saving run data to dir: {run_dir}")
#%%

save_epoch_interval = 20

learning_rate = 1e-4
epochs = 400
batch_size = 256

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
if not os.path.exists(trainset_path):
    print("Dataset not found. Check if path is correct")
    exit()
else:
    print(f"Dataset found. Loading dataset from {trainset_path}")
# %%
train_set = SpectrogramFileDataset(trainset_path, transform=ToTensor())
test_set = SpectrogramFileDataset(testset_path, transform=ToTensor())
print(f"{'#'*3} {'Dataset info' + ' ':{'#'}<{24}}")
print(f"Total dataset length: {len(train_set) + len(test_set)}")

#%%
# train_size = int(len(spectrogram_dataset) * (1 - val_percentage))
# test_size = len(spectrogram_dataset) - train_size #int(len(spectrogram_dataset) * val_percentage)
# train_set, val_set = torch.utils.data.random_split(spectrogram_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))

# %%
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=12)
#%%
print(f"Train length: {len(train_set)} Test length: {len(test_set)}")
print(f"Train batch num: {len(train_loader)} Test batch num: {len(test_loader)}")
# %%
encoder = VariationalEncoder()
decoder = VariationalDecoder()
model = VariationalAutoEncoder(encoder, decoder)

writer.add_graph(model, next(iter(train_loader))[0]) # add model to tensorboard and

#codes = dict(mu = list(), variance = list(), y=list())
#%%
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_epoch = 0

if continue_training:
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')] #parse checkpoint files
    checkpoint_epochs = [int(f.split('_e')[-1].split('.')[0]) for f in checkpoint_files] #get the epoch number from the checkpoint file name
    newest_checkpoint_file = checkpoint_files[np.argmax(checkpoint_epochs)] #get the newest checkpoint file

    checkpoint = torch.load(os.path.join(model_dir, newest_checkpoint_file))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'#'*3} {'Training info' + ' ':{'#'}<{24}}")
print(f"Using {device} device for training")
model.to(device)
model.train()
#%%
print(f"{'#'*3} {'Training' + ' ':{'#'}<{24}}")
for epoch in range(start_epoch, epochs):
    train_loss = 0
    test_loss = 0

    train_samples_count = 0
    if epoch > 0:
        model.train()
        for batch, _, _ in train_loader:

            batch = batch.to(device)
            x_hat, mu, logvar = model(batch)

            loss = loss_function(x_hat, batch, mu, logvar)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_samples_count += len(batch)
            print(f"epoch: {epoch:5} | train loss: {train_loss / train_samples_count:16.6f} | test loss: {test_loss / len(test_loader.dataset):16.6f}", end = "\r")

        writer.add_scalar("Loss/train", train_loss / len(train_loader.dataset), epoch)

    # print(f"tain set length: {len(train_loader.dataset)} train_samples_count: {train_samples_count}")

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

    if epoch != 0 and epoch % save_epoch_interval == 0:
        torch.save( {
            "epoch" : epoch,
            "model_state_dict" :model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "loss" : train_loss / len(train_loader.dataset),

        }, f"{model_dir}/checkpoint_e{epoch}.pt")
        #torch.save(optimizer.state_dict(), f"{model_dir}/optimizer_{epoch}.pt")

    print()
#%%
# time_mean_width =  np.mean([event.end_time_index - event.start_time_index for event in all_sample_events])
# time_std_with = np.std([event.end_time_index - event.start_time_index for event in all_sample_events])
# time_mean_width, time_std_with