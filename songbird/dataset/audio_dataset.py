# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import os, sys
import warnings
import songbird.audio.audio_processing as ap
import numpy as np

from torch.utils.data import Dataset

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