import pandas as pd
import datetime
import time
import json
import re
import os
import shutil 

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
def get_associated_birds(associated_taxa, id_of_sample):
    #check if associated_taxa string is empty, None or nan
    if not associated_taxa or associated_taxa is None or pd.isnull(associated_taxa):
        return []
    associated_taxa = associated_taxa.replace('has background sounds:', '')
    split_taxa = [taxa.strip().lower() for taxa in associated_taxa.split('|')]
    #species_key_dict is a global variable
    bird_species_keys = [species_to_key_dict[taxa] for taxa in split_taxa if species_to_key_dict.get(taxa) is not None]

    for key in bird_species_keys:
        species_samples_info[key]['background_sample_ids'].append(id_of_sample)

    return bird_species_keys

def get_formatted_date(date):
    if not date or date is None or pd.isnull(date):
        return None
    try:
        return datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime("%Y-%m-%d")
    except:
        return None

def get_audio_recording_info(gbif_id):
    link = multimedia_df.at[gbif_id, 'identifier']
    time_sec = multimedia_df.at[gbif_id, 'description']
    audio_format = multimedia_df.at[gbif_id, 'format']

    link = link if not pd.isnull(link) else None
    time_sec = int(re.sub(r"\D", "", time_sec)) if not pd.isnull(time_sec) else None # Remove anything other than digits and convert the digits to int value
    audio_format = audio_format.replace('audio/', '') if not pd.isnull(audio_format) else None #remove audio/ from format description
    audio_format += '/mp3'
    return link, time_sec, audio_format

def get_species_info(species_key):
    species_rows = occurrence_df[occurrence_df['speciesKey'] == species_key]
    genus_key = species_rows['genusKey'].unique()
    scientific_name = species_rows['species'].unique()
    common_name = species_rows['vernacularName'].unique()
    if len(genus_key) > 1:  # check if there is more than one genus key for species key
        print("WARNING: species with key: {0} has multiple 'genus keys': {1}! Picking the first one!".format(species_key, genus_key))
    if len(scientific_name) > 1: # check if there is more than one scientific name for species key
        print("WARNING: species with key: {0} has multiple 'scientific names': {1}! Picking the first one!".format(species_key, scientific_name))
    if len(common_name) > 1: # check if there is more than one common name for species key
        print("WARNING: species with key: {0} has multiple 'common names': {1}! Picking the first one!".format(species_key, common_name))
    return genus_key[0], scientific_name[0], common_name[0]

def add_sample_dict(species_samples_dict, forefront_bird_key, gbif_id, decimal_latitude, decimal_longitude, event_date, behavior, associated_taxa, progress_bar):
    link, time_sec, audio_format = get_audio_recording_info(gbif_id)
    gbif_id_int = int(gbif_id)
    species_samples_info[forefront_bird_key]['forefront_sample_ids'].append(gbif_id_int)

    species_samples_dict[gbif_id_int] = {
        'gbifID' : gbif_id_int,
        'recording_link' : link,
        'recording_time_sec' : time_sec,
        'audio_format' : audio_format,
        'decimal_latitude' : float(decimal_latitude) if not pd.isnull(decimal_latitude) else None,
        'decimal_longitude': float(decimal_longitude)if not pd.isnull(decimal_longitude) else None,
        'date' : get_formatted_date(event_date),
        'behavior' : get_behavior(behavior),
        'forefront_bird_key' : forefront_bird_key,
        'background_birds_keys' : get_associated_birds(associated_taxa, gbif_id_int),
    }
    progress_bar.progress()
    progress_bar.print()


this_file_dir_path = os.path.dirname(os.path.abspath(__file__))

gbif_path = os.path.join(this_file_dir_path, 'xeno_canto_bsfatw') #absolute path to xeno canto meta data 
#load relevant txt files as pandas dfs
occurrence_df = pd.read_csv(os.path.join(gbif_path, 'occurrence.txt'), sep = '\t')
multimedia_df = pd.read_csv(os.path.join(gbif_path, 'multimedia.txt'), sep = '\t')

#Check for columns with no values and delete them
occurrence_columns_with_no_values = []
for column in occurrence_df.columns:
    if occurrence_df[column].isnull().all():
        occurrence_columns_with_no_values.append(column)
print("Dropping empty columns of occurrence_df")
occurrence_df.drop(occurrence_columns_with_no_values, axis=1, inplace=True)

#drop rows that do not reperesent audio files 
image_link_indices = multimedia_df[multimedia_df['format'] != 'audio/mpeg'].index
multimedia_df.drop(image_link_indices, inplace = True)
#Check for columns with no values and delete them
multimedia_columns_with_no_values = []
for column in multimedia_df.columns:
    if multimedia_df[column].isnull().all():
        multimedia_columns_with_no_values.append(column)
print("Dropping empty columns of multimedia_df")
multimedia_df.drop(multimedia_columns_with_no_values, axis=1, inplace=True)
#set df index to gbifID
multimedia_df.set_index('gbifID', inplace = True)

#Define all undefined species as unknown species
occurrence_df['species'] = occurrence_df['species'].fillna('Unkown').str.lower()
occurrence_df['vernacularName'] = occurrence_df['vernacularName'].fillna('Unkown').str.lower()
occurrence_df['speciesKey'] = occurrence_df['speciesKey'].fillna(0).astype(int)
occurrence_df['genusKey'] = occurrence_df['genusKey'].fillna(0).astype(int)

#speciesKey	species
species_name_key_df = occurrence_df[['speciesKey', 'species']]
species_to_key_dict = dict(zip(species_name_key_df['species'], species_name_key_df['speciesKey']))

#Define Species info data
species_info_columns = ['common_name','scientific_name', 'species_key', 'genus_key', 'forefront_recordings', 'background_recordings']
species_info_arr = []
#Create sample info
data_dictionary_path = os.path.join(this_file_dir_path, 'dataset', 'data_dictionary')
#check if data_dictionary directory exists
if os.path.exists(data_dictionary_path):
    for filename in os.listdir(data_dictionary_path):
        file_path = os.path.join(data_dictionary_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete {}. Exceptions: {}'.format(file_path, e)) 
else:
    os.makedirs(data_dictionary_path)

samples_metadata_path = os.path.join(data_dictionary_path, 'samples_metadata.json')
samples_per_species = occurrence_df.groupby(['speciesKey']) #group samples of each species
progress_bar = ProgressBar(len(occurrence_df))
samples_metadata = {}
species_samples_info = { species_key : { 'forefront_sample_ids': [], 'background_sample_ids': [] } for species_key in samples_per_species.groups.keys()}

for species_key, samples in samples_per_species:    #iterate over each species
    genus_key, scientific_name, common_name = get_species_info(species_key)
    #add species info 'row'
    species_info_arr.append([
        common_name,
        scientific_name, 
        species_key, 
        genus_key, 
        len(samples), 
        0
    ])

    samples.apply(lambda row: add_sample_dict(samples_metadata, species_key, row['gbifID'], row['decimalLatitude'], row['decimalLongitude'], row['eventDate'], row['behavior'], row['associatedTaxa'], progress_bar), axis=1)

with open(samples_metadata_path, 'w') as outfile:
    json.dump(samples_metadata, outfile, indent=4)
#write species samples info 
species_info_path = os.path.join(data_dictionary_path, 'species_sample_info.json')
with open(species_info_path, 'w') as outfile:
    json.dump(species_samples_info, outfile, indent=4)
#Create dataframe for species info
species_info_df = pd.DataFrame(data = species_info_arr, columns = species_info_columns)
species_info_path = os.path.join(data_dictionary_path, 'species_info.csv')
species_info_df['background_recordings'] = [len(species_samples_info[key].get('background_sample_ids', [])) for key in species_info_df['species_key']]
#clean up unknown species row
species_info_df.loc[species_info_df['species_key'] == 0, 'common_name'] = 'unknown'
species_info_df.loc[species_info_df['species_key'] == 0, 'genus_key'] = 0

species_info_df.to_csv(species_info_path, index=False)

print()
print("data dictionary creation completed")


