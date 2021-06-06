import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import songbird.variational_autoencoder.variational_autoencoder as vae
from songbird.variational_autoencoder.variational_autoencoder import loss_function, mse_loss_function
import songbird.audio_events.audio_processing as ap
#from songbird.audio_events.audio_processing import load_audio_sample
import pydub
from pydub import AudioSegment
from pydub.playback import play

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib
from matplotlib import pyplot as plt

import warnings
import numpy as np
import scipy.io.wavfile as wavfile

import scipy

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        self.dense1 = nn.Linear(4410, 400) 
       # self.dense2 = nn.Linear(256, 128) 
        self.mean_dense = nn.Linear(400, 400)
        self.variance_dense = nn.Linear(400, 400)

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
        self.dense1 = nn.Linear(400, 400)
        #self.dense2 = nn.Linear(128, 256) 
        self.dense3 = nn.Linear(400, 4410) 

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        #x =  F.relu(self.dense2(x))
        x =  F.tanh(self.dense3(x))
        return x  

# samples_arr = load_samples(sample_ids, sample_rate)
# samples_events_arr = load_sample_events(sample_ids)
# samples_events_arr = [[{'start' : ap.unit_time_to_index(event["start_sec"],  sample_rate), "end" : ap.unit_time_to_index(event["end_sec"],  sample_rate) } for event in sample_events] for sample_events in samples_events_arr]


# audio_event = get_sample_audio_event(samples_arr, samples_events_arr, 0, 1, int(sample_rate * 0.5))
# buffer = int(sample_rate * 0.5)   
# sample_arr = samples_arr[0][ samples_events_arr[0][1]['start'] - buffer : samples_events_arr[0][1]['end'] + buffer ]


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

class AudioDataset(Dataset):
    def __init__(self, audio_ids, audio_dir, sample_rate, sample_dim, step_size):
        self.sample_rate = sample_rate

        self.samples = []

        for audio_id in audio_ids:
            audio_arr = ap.load_audio_sample(audio_id, sample_rate, audio_dir)[0]
            for index in range(0, len(audio_arr), step_size):
                end_index = index + sample_dim
                if end_index >= len(audio_arr): #while the end_index oversteps the sample length
                    if index == 0:
                        warnings.warn(f"Warning! Audio sample with id {audio_id} was not long enough to be included in the data!\nSample dim: {sample_dim} | Audio length: {len(audio_arr)}")
                    break
                self.samples.append(audio_arr[index:end_index])

        self.samples = np.asarray(self.samples)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], 0


def set_torch_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def train_plus(model, train_loader, test_loader, loss_fn, learning_rate, epochs, plot_fn = None, plot_params = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device for training")
    model.to(device)
    model.train()

    #codes = dict(mu = list(), variance = list(), y=list())
    #input processing variables
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #general variables
    for epoch in range(0, epochs + 1):
        if epoch > 0:
            model.train()
            train_loss = 0
            for x, _ in train_loader:

                x = x.to(device)
                x_hat, mu, logvar = model(x)
                loss = loss_fn(x_hat, x, mu, logvar)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"epoch: {epoch:4} | train loss: {train_loss / len(train_loader.dataset):10.6f}", end = "\r")

        plot_x = None
        plot_x_hat = None
        means, variance, labels = list(), list(), list()
        with torch.no_grad():
            model.eval()
            test_loss = 0
            for x, y in test_loader:
                    
                x = x.to(device)
                x_hat, mu, logvar = model(x)
    
                test_loss += loss_fn(x_hat, x, mu, logvar)

                means.append(mu.detach())
                variance.append(logvar.detach())
                labels.append(y.detach())

                if len(x) > plot_params["plot_num"]:
                    plot_x = x
                    plot_x_hat = x_hat
        
        plot_fn(plot_x, plot_x_hat, f'Epoch_{epoch}', 1, **plot_params)
    
    return model

def train(model, sample_arr, model_dir):
    
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device for training")
    model.to(device)

    model.train()

    epochs = 20000
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #input processing variables
    model_input_dim = 4410
    batch_size = 1
    step_size = 128
    #general variables
    audio_sample_length = len(sample_arr)
    
    for epoch in range(epochs):
        model.train() 
        train_loss = 0   #total epoch mse error 
        for index in range(0, audio_sample_length, step_size * batch_size): #iterate over sample arr (currently does not do overlapping sampling)
            end_index = index + step_size * ( batch_size - 1 ) + model_input_dim #get the end index of the train data slice
            while end_index >= audio_sample_length: #while the end_index oversteps the sample length
                end_index -= step_size #subtract the step size
            if end_index < index + model_input_dim: #if the reduced train data slice is smaller index then the model input 
                continue #skip training step
            
            batch_x = torch.tensor(sample_arr[index:end_index]) #get train data slice
            batch_x = batch_x.to(device).unfold(0, model_input_dim, step_size) #move data to device and roll data into batched samples
            #print(audio_window.shape)
            y_pred, mean, variance = model(batch_x) #get output of batch
            loss = vae.loss_function(y_pred, batch_x, mean, variance) #get reconstruction error
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch: {epoch:4} | progress: {index/audio_sample_length*100:.2f}% | train loss: {loss.item():10.6f}", end = "\r")
        
        print(f"epoch: {epoch:4} | progress: {100:.2f}% | train mse: {train_loss:10.6f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "vae" + ".pt"))

    torch.save(model.state_dict(), os.path.join(model_dir, "vae" + ".pt"))
    
    return model

def encode_decode_sample(variational_auto_encoder, sample_arr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device for inference")
    variational_auto_encoder.to(device)
    with torch.no_grad():
        variational_auto_encoder.eval()

        model_input_dim = 4410
        batch_size = 1
        window_step_size = 4410
        audio_sample_length = len(sample_arr)

        decoded_audio = []
        for index in range(0, audio_sample_length, window_step_size):
            max_index = index + model_input_dim
            if max_index >= audio_sample_length:
                continue
            
            audio_window = torch.tensor(sample_arr[index:max_index]).to(device)
            y_pred, _, _ = variational_auto_encoder(audio_window)

            decoded_audio.append(y_pred.to('cpu').detach().numpy())

        decoded_audio = np.concatenate(decoded_audio).ravel()

    return decoded_audio

def normalize(audio_arr):
    return (audio_arr + 1) / 2

def denormalize(audio_arr):
    return audio_arr * 2 - 1

def load_samples(sample_ids, sample_rate, sample_dir = None):
    return [ap.load_audio_sample(sample_id, sample_rate, sample_dir)[0] for sample_id in sample_ids]

def load_sample_events(sample_ids):
    return [ap.load_audio_events(sample_id) for sample_id in sample_ids]

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

def show_mnist(x, y, label=None, fig_num = 1, plot_num = 4, report_dir = None):
    for fig_index in range(fig_num):
        fig, axs = plt.subplots(2, 4)
        in_pic = x.data.cpu().view(-1, 28, 28)
        #axs.suptitle(label + ' - real test data / reconstructions', color='w', fontsize=16)
        for i in range(plot_num):
            axs[0, i].imshow(in_pic[fig_index * plot_num + i])#plt.subplot(1, plot_num, i + 1)
            axs[0, i].axis('off')
            #plt.imshow(in_pic[i+plot_num+plot_index])
            #plt.axis('off')
        out_pic = y.data.cpu().view(-1, 28, 28)
        #plt.figure(figsize=(18,6))
        for i in range(plot_num):
            #plt.subplot(1, plot_num, i + 1)
            axs[1, i].imshow(out_pic[fig_index * plot_num + i])
            axs[1, i].axis('off')

        fig.canvas.set_window_title(f"{label} - real test data / reconstructions'")
        plt.savefig(os.path.join(report_dir, f'{label}_{fig_index}.png'))

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
         
def train_on_mnist():

    class MnistEncoder(nn.Module):
        def __init__(self):
            super(MnistEncoder, self).__init__()
            
            #self.conv1 = nn.Conv1d(1, 6, 3)
            self.dense1 = nn.Linear(784, 400) 
        # self.dense2 = nn.Linear(256, 128) 
            self.mean_dense = nn.Linear(400, 20)
            self.variance_dense = nn.Linear(400, 20)

        def forward(self, x):
            x =  F.relu(self.dense1(x))
            #x =  F.relu(self.dense2(x))
            x_mean =  self.mean_dense(x)
            x_variance =  self.variance_dense(x)
            return  x_mean, x_variance

    class MnistDecoder(nn.Module):
        def __init__(self):
            super(MnistDecoder, self).__init__()
            
            #self.conv1 = nn.Conv1d(1, 6, 3)
            self.dense1 = nn.Linear(20, 400)
            #self.dense2 = nn.Linear(128, 256) 
            self.dense3 = nn.Linear(400, 784) 

        def forward(self, x):
            x =  F.relu(self.dense1(x))
            #x =  F.relu(self.dense2(x))
            x =  F.sigmoid(self.dense3(x))
            return x  




    model_name = "mnist_model"
    report_dir = os.path.join(ap.project_base_dir, "reports", "vae", model_name)
    ap.empty_or_create_dir(report_dir)
    
    set_torch_seeds(42)

    batch_size = 256
    kwargs = { 'num_workers': 1, 'pin_memory': True }

    train_loader = torch.utils.data.DataLoader(
        MNIST(
            './data',
            train=True, 
            download=True, 
            transform= transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Lambda(lambda x: torch.flatten(x))
                ]
            )
        ),
        batch_size=batch_size,
        shuffle=True, 
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        MNIST(
            './data', 
            train=False, 
            download=True, 
            transform=transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Lambda(lambda x: torch.flatten(x))
                ]
            )
        ),
        batch_size=batch_size, shuffle=True, **kwargs)


    encoder = MnistEncoder()
    decoder = MnistDecoder()
    model = vae.VariationalAutoDecoder(encoder, decoder)

    plot_params = {
        "report_dir" : report_dir,
        "plot_num" : 4
    }


    learning_rate = 1e-4
    epochs = 100

    train_plus(model, train_loader, test_loader, loss_function, learning_rate, epochs, show_mnist, plot_params)

    print()
    print()

def train_on_bird_sounds():
    model_name = "songbird_model"
    sample_ids = ["2243804495"]
    sample_rate = 44100

    model_dir = os.path.join(ap.project_base_dir, "models", "vae")
    data_dir = ap.data_dir
    audio_dir = os.path.join(ap.data_dir, "raw")
    generated_dir = os.path.join(data_dir, "generated")
    report_dir = os.path.join(ap.project_base_dir, "reports", "vae", model_name)
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    ap.empty_or_create_dir(generated_dir)
    ap.empty_or_create_dir(report_dir)

    set_torch_seeds(42)

    dataset = AudioDataset(sample_ids, audio_dir, sample_rate, 4410, 128)

    print(len(dataset))
    kwargs = { 'num_workers': 1, 'pin_memory': True }
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, **kwargs)
    #test_loader = DataLoader(dataset, batch_size=64, shuffle=True, **kwargs)

    encoder = VariationalEncoder()
    decoder = VariationalDecoder()
    model = vae.VariationalAutoDecoder(encoder, decoder)

    print(model)
    
    plot_params = {
        "report_dir" : report_dir,
        "plot_num" : 4,
        "sample_rate": sample_rate
    }

    learning_rate = 1e-4
    epochs = 100

    #sample_arr = normalize(sample_arr)

    train_plus(model, train_loader, train_loader, mse_loss_function, learning_rate, epochs, show_audio, plot_params)

    # sample_arr = encode_decode_sample(variational_auto_encoder, sample_arr)

    # sample_arr = denormalize(sample_arr)

    # wavfile.write(os.path.join(generated_dir, sample_id + '.wav'), sample_rate, sample_arr)

if __name__ == "__main__":
    train_on_bird_sounds()
    # train_on_mnist()



