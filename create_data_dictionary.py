import pandas as pd
import datetime
import time
import json
import re
import os

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
        #hour, min, sec = self.mean_time_sec.strftime("%H %M %S").split()
        print(self.bar_string.format('#' * int(self.progress_count / self.number_of_samples * self.bar_length) , self.progress_count, self.number_of_samples, hr, min, sec), end="\r", flush=True )

def unclear_behavior(behavior):
    if '?' in behavior or not behavior:
        return True
    if "uncertain" in behavior:
        return True
    #check if behavior has only alphabetical letters or spaces 
    if not all(char.isalpha() or char.isspace() for char in behavior):
        return True
        
    return False

def get_behavior(behaviors):
    #check if behaviors string is empty, None or nan
    if not behaviors or behaviors is None or pd.isnull(behaviors):
        return []
    #seperate sample behaviors and append to array
    split_behaviors = [behavior.strip().lower() for behavior in behaviors.split(',')] # remove leading and trailinf white spaces and set to lower case
    #remove double quotes
    split_behaviors = [behavior.replace('"', '') for behavior in split_behaviors]
    #remove single quotes    
    split_behaviors = [behavior.replace("'", '') for behavior in split_behaviors]

    return [behavior for behavior in split_behaviors if not unclear_behavior(behavior)]

'''
sample associated taxa:

has background sounds: Phylloscopus trochilus|Corvus corone|Dendrocopos major|Turdus philomelos|Cyanistes caeruleus
has background sounds: Turdus iliacus|Turdus pilaris|Numenius arquata|Corvus cornix|Emberiza citrinella
'''
def get_associated_birds(associated_taxa):
    #check if associated_taxa string is empty, None or nan
    if not associated_taxa or associated_taxa is None or pd.isnull(associated_taxa):
        return []
    associated_taxa = associated_taxa.replace('has background sounds:', '')
    split_taxa = [taxa.strip().lower() for taxa in associated_taxa.split('|')]
    #species_key_dict is a global variable
    bird_species_keys = [species_to_key_dict[taxa] for taxa in split_taxa if species_to_key_dict.get(taxa) is not None]

    for key in bird_species_keys:
        if key in species_background_occurrence_dict:
            species_background_occurrence_dict[key] += 1
        else:
            species_background_occurrence_dict[key] = 1

    return bird_species_keys

def get_formatted_date(date):
    if not date or date is None or pd.isnull(date):
        return None
    try:
        return datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime("%Y-%m-%d")
    except:
        return None

def get_audio_recording_info(gbif_id):
    audio_info = multimedia_df.loc[(multimedia_df['gbifID'] == gbif_id) & (multimedia_df['format'] == 'audio/mpeg') ]
    link =  None if pd.isnull(audio_info.iloc[0]['identifier']) else audio_info.iloc[0]['identifier']
    time_sec = None if pd.isnull(audio_info.iloc[0]['description']) else int(re.sub("\D", "", audio_info.iloc[0]['description']))
    audio_format = None if pd.isnull(audio_info['format'].iloc[0]) else audio_info['format'].iloc[0].replace('audio/', '')
    return link, time_sec, audio_format

def get_species_info(species_key):
    species_rows = occurrence_df[occurrence_df['speciesKey'] == species_key]
    genus_key = species_rows['genusKey'].unique()
    scientific_name = species_rows['species'].unique()
    if len(genus_key) > 1:
        print("WARNING: species with key: {0} has multiple 'genus keys': {1}".format(species_key, genus_key))
    if len(scientific_name) > 1:
        print("WARNING: species with key: {0} has multiple 'scientific names': {1}".format(species_key, scientific_name))
    return genus_key[0], scientific_name[0]

def add_sample_dict(species_samples_dict, gbif_id, decimal_latitude, decimal_longitude, event_date, behavior, associated_taxa):
    link, time_sec, audio_format = get_audio_recording_info(gbif_id)
    gbif_id_int = int(gbif_id)
    species_samples_dict[gbif_id_int] = {
        'gbifID' : gbif_id_int,
        'recording_link' : link,
        'recording_time_sec' : time_sec,
        'audio_format' : audio_format,
        'decimal_latitude' : float(decimal_latitude) if not pd.isnull(decimal_latitude) else None,
        'decimal_longitude': float(decimal_longitude)if not pd.isnull(decimal_longitude) else None,
        'date' : get_formatted_date(event_date),
        'behavior' : get_behavior(behavior),
        'background_birds' : get_associated_birds(associated_taxa),
    }


gbif_path = 'xeno_canto_bsfatw'

occurrence_df = pd.read_csv(os.path.join(gbif_path, 'occurrence.txt'), sep = '\t')
multimedia_df = pd.read_csv(os.path.join(gbif_path, 'multimedia.txt'), sep = '\t')


occurrence_columns_with_no_values = []
for column in occurrence_df.columns:
    if occurrence_df[column].isnull().all():
        occurrence_columns_with_no_values.append(column)
print("Dropping empty columns of occurrence_df")
occurrence_df.drop(occurrence_columns_with_no_values, axis=1, inplace=True)

image_link_indices = multimedia_df[multimedia_df['format'] != 'audio/mpeg'].index


multimedia_columns_with_no_values = []
for column in multimedia_df.columns:
    if multimedia_df[column].isnull().all():
        multimedia_columns_with_no_values.append(column)
print("Dropping empty columns of multimedia_df")
multimedia_df.drop(multimedia_columns_with_no_values, axis=1, inplace=True)

#Define all undefined species as unknown species
occurrence_df['species'] = occurrence_df['species'].fillna('Unkown').str.lower()
occurrence_df['speciesKey'] = occurrence_df['speciesKey'].fillna(0).astype(int)
occurrence_df['genusKey'] = occurrence_df['genusKey'].fillna(0).astype(int)

#speciesKey	species
species_name_key_df = occurrence_df[['speciesKey', 'species']]
species_to_key_dict = dict(zip(species_name_key_df['species'], species_name_key_df['speciesKey']))

species_background_occurrence_dict = {}

#Create Species info
species_info_columns = ['common_name','scientific_name', 'species_key', 'genus_key', 'forefront_recordings', 'background_recordings']
#species_info_df = pd.DataFrame(columns = species_info_columns)
species_info_arr = []
#Create sample info
data_dictionary_path = os.path.join('dataset', 'data_dictionary')

samples_info_path = os.path.join(data_dictionary_path, 'samples_info')
samples_per_species = occurrence_df.groupby(['speciesKey']) #group samples of each species
progress_bar = ProgressBar(len(occurrence_df))
for species_key, samples in samples_per_species:    #iterate over each species

    species_samples_path = os.path.join(samples_info_path, str(species_key) + '.json')

    genus_key, scientific_name = get_species_info(species_key)

    # species_info_arr.append({
    #     'common_name' : None,
    #     'scientific_name' : scientific_name, 
    #     'species_key' : species_key, 
    #     'genus_key' : genus_key, 
    #     'forefront_recordings' : len(samples), 
    #     'background_recordings' : 0
    # })

    species_info_arr.append([
        None,
        scientific_name, 
        species_key, 
        genus_key, 
        len(samples), 
        0
    ])

    data_dict = {
        'species_key' : int(species_key),
        'genus_key' : int(genus_key),
        'scientific_name' : scientific_name,
        'common_name' : None,
        'samples' : {} #create and array of sample dicts
    }

    for index, sample in samples.iterrows():    #iterate over each sample of the species
        gbifID = sample['gbifID']
        link, time_sec, audio_format = get_audio_recording_info(gbifID)
        data_dict['samples'][int(gbifID)] = {
            'gbifID' : int(gbifID),
            'recording_link' : link,
            'recording_time_sec' : time_sec,
            'audio_format' : audio_format,
            'decimal_latitude' : float(sample['decimalLatitude']) if not pd.isnull(sample['decimalLatitude']) else None,
            'decimal_longitude': float(sample['decimalLongitude'])if not pd.isnull(sample['decimalLongitude']) else None,
            'date' : get_formatted_date(sample['eventDate']),
            'behavior' : get_behavior(sample['behavior']),
            'background_birds' : get_associated_birds(sample['associatedTaxa']),
        }
        progress_bar.progress()
        progress_bar.print()

    with open(species_samples_path, 'w') as outfile:
        json.dump(data_dict, outfile, indent=4)

species_info_df = pd.DataFrame(data = species_info_arr, columns = species_info_columns)
species_info_path = os.path.join(data_dictionary_path, 'species_info.csv')
species_info_df['background_recordings'] = [species_background_occurrence_dict.get(key, 0) for key in species_info_df['species_key']]
species_info_df.to_csv(species_info_path, index=False)


