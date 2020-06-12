import pandas as pd
import json
import datetime
import argparse
import os
import requests
#import request

class ProgressBar:
    def __init__(self, number_of_samples, bar_length = 50):
        self.number_of_samples = number_of_samples
        self.bar_length = bar_length
        self.progress_count = 0 
        self.bar_string = '> [{0:' + str(bar_length) + '}] | {1}/{2} | etc: {3:.0f}:{4:.0f}:{5:.0f}'
        
        self.start_time = None
        self.mean_time_sec = None
        self.progress_initiated = False

    def progress(self, increase_by = 1):
        self.progress_count += increase_by
        if not self.progress_initiated:
            self.progress_initiated = True
            self.start_time = datetime.datetime.now()
        self.mean_time_sec = (datetime.datetime.now() - self.start_time).total_seconds() / self.progress_count * (self.number_of_samples - self.progress_count)

    def print(self):
        min, sec = divmod(self.mean_time_sec, 60)
        hr, min = divmod(min, 60)
        print(self.bar_string.format('#' * int(self.progress_count / self.number_of_samples * self.bar_length) , self.progress_count, self.number_of_samples, hr, min, sec), end="\r", flush=True )

def load_species_sample_info(samples_info_path, species_key):
    samples_info_path = os.path.join(samples_info_path, str(species_key) + '.json')
    with open(samples_info_path) as json_file:
        data = json.load(json_file)
    return data

def download_species_samples(species_sample_dict, destination_path, limit = None):
    samples_dict = species_sample_dict['samples']
    samples_path = os.path.join(destination_path, str(species_sample_dict['species_key']))

    if not os.path.exists(samples_path):
        os.mkdir(samples_path)

    count = 0
    for gbif_id, sample_info in samples_dict.items():
        file_path = os.path.join(samples_path, str(gbif_id) + '.mpeg')
        url_link = sample_info['recording_link']

        r = requests.get(url_link)

        with open(file_path, 'wb') as f:
            f.write(r.content)
        
        count += 1
        if (limit <= count):
            break

parser = argparse.ArgumentParser(description='Download mpeg samples from species described int the data_dictionary directory')
parser.add_argument('--sample_min', type=int, required=False, default=0,
                        help='set the minimum forefront sample amount a species should have')
parser.add_argument('--download_max', type=int, required=False, default=0,
                        help='set the maximum forefront samples that should be downloaded for each species should have')

args = parser.parse_args()


dataset_path = 'dataset'
raw_data_path = os.path.join(dataset_path, 'raw')

data_dictionary_path = os.path.join(dataset_path, 'data_dictionary')
samples_info_path = os.path.join(data_dictionary_path, 'samples_info')
species_info_path = os.path.join(data_dictionary_path, 'species_info.csv')

species_info_df = pd.read_csv(species_info_path)

not_meeting_requirements_indices = species_info_df[species_info_df['forefront_recordings'] < args.sample_min].index
species_info_df.drop(not_meeting_requirements_indices, inplace = True)
print("{} species have more than {} samples".format(len(species_info_df), args.sample_min))

species_sample_data = load_species_sample_info(samples_info_path, 2473663)
download_species_samples(species_sample_data, raw_data_path, 1)
#rint(species_sample_data)

#print(species_info_df.head())

