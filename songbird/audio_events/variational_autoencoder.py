import audio_processing as ap
#from songbird.audio_events.audio_processing import load_audio_sample
import pydub
from pydub import AudioSegment
from pydub.playback import play

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.io.wavfile as wavfile
import os

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        self.dense1 = nn.Linear(512, 256) 
        self.dense2 = nn.Linear(256, 128) 
        self.dense3 = nn.Linear(128, 64)

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        x =  F.relu(self.dense2(x))
        x =  self.dense3(x)
        return x

class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 256) 
        self.dense3 = nn.Linear(256, 512) 

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        x =  F.relu(self.dense2(x))
        x =  self.dense3(x)
        return x  

class VariationalAutoDecoder(nn.Module):  
    def __init__(self, encoder, decoder):
        super(VariationalAutoDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_variational_auto_encoder(variational_auto_encoder, sample_arr, model_dir):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device for training")
    variational_auto_encoder.to(device)
    #training variables and functions
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(variational_auto_encoder.parameters(), lr=learning_rate)
    #input processing variables
    model_input_dim = 512
    batch_size = 64
    step_size = 128
    epoch_num = 200
    #general variables
    audio_sample_length = len(sample_arr)
    
    for epoch in range(epoch_num):
        epoch_mse = 0
        for index in range(0, audio_sample_length, step_size * batch_size): #iterate over sample arr (currently does not do overlapping sampling)
            end_index = index + step_size * ( batch_size - 1 ) + model_input_dim #get the end index of the train data range
            while end_index >= audio_sample_length: #while the end_index oversteps the sample length
                end_index -= step_size #subtract the step size
            if end_index < index + model_input_dim: #if the reduced train data range is smaller index then the model input 
                continue #skip training step
            
            batch_x = torch.tensor(sample_arr[index:end_index]) #get batch size
            batch_x = batch_x.to(device).unfold(0, model_input_dim, step_size)
            #print(audio_window.shape)
            y_pred = variational_auto_encoder(batch_x)
            loss = loss_fn(y_pred, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_mse += loss.item()
            print(f"epoch: {epoch:4} | progress: {index/audio_sample_length*100:.2f}% | mse loss: {loss.item():10.6f}", end = "\r")
        
        print(f"epoch: {epoch:4} | progress: {100:.2f}% | epoch mse: {epoch_mse:10.6f}")
        if epoch % 10 == 0:
            torch.save(variational_auto_encoder.state_dict(), os.path.join(model_dir, "vae" + ".pt"))

    torch.save(variational_auto_encoder.state_dict(), os.path.join(model_dir, "vae" + ".pt"))
    
    return variational_auto_encoder

def encode_decode_sample(variational_auto_encoder, sample_arr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_input_dim = 512
    batch_size = 1
    window_step_size = 512
    audio_sample_length = len(sample_arr)

    decoded_audio = []
    for index in range(0, audio_sample_length, window_step_size):
        max_index = index + model_input_dim
        if max_index >= audio_sample_length:
            continue
        
        audio_window = torch.tensor(sample_arr[index:max_index]).to(device)
        y_pred = variational_auto_encoder(audio_window)

        decoded_audio.append(y_pred.to('cpu').detach().numpy())

    decoded_audio = np.concatenate(decoded_audio).ravel()

    return decoded_audio

if __name__ == "__main__":
    sample_id = "2243804495"
    sample_rate = 44100

    data_dir = ap.data_dir
    generated_dir = os.path.join(data_dir, "generated")
    model_dir = os.path.join(ap.project_base_dir, "models", "vae")
    ap.empty_or_create_dir(generated_dir)



    sample_arr, _ = ap.load_audio_sample(sample_id, sample_rate)

    audio_segment = pydub.AudioSegment(
        sample_arr.tobytes(), 
        frame_rate=sample_rate,
        sample_width=sample_arr.dtype.itemsize, 
        channels=1
    )

    #play(audio_segment)

    torch.manual_seed(42)

    encoder = VariationalEncoder()
    decoder = VariationalDecoder()
    variational_auto_encoder = VariationalAutoDecoder(encoder, decoder)

    print(variational_auto_encoder)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    variational_auto_encoder = train_variational_auto_encoder(variational_auto_encoder, sample_arr, model_dir)

    sample_arr = encode_decode_sample(variational_auto_encoder, sample_arr)

    wavfile.write(os.path.join(generated_dir, sample_id + '.wav'), sample_rate, sample_arr)
