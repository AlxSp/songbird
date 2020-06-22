import pandas as pd
import json
import datetime
import argparse
import os
import requests
import shutil
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

def get_species_sample_ids(species_sample_info : dict, species_key_list : list):
    return {key: species_sample_info[key] for key in species_key_list}

def filter_species_samples(species_sample_ids : dict, include_background_samples : bool = False, sample_limit : int = None, already_downloaded_sample_ids : set = ()):
    '''
    uses the species_sample_ids dict 
    '''
    #for each species trim the samples to the sample download limit
    for species_key, sample_dict in species_sample_ids.items():
        species_samples_num = 0
        #if sample download limit is set
        if sample_limit is not None:
            if len(sample_dict['forefront_sample_ids']) >= sample_limit: #if there are more forefront samples than the limit
                sample_dict['forefront_sample_ids'] = sample_dict['forefront_sample_ids'][0:sample_limit] #get number of forefront samples according to the limit
                sample_dict['background_sample_ids'] = [] #set background samples to 0
                species_samples_num = sample_limit
            elif include_background_samples:    #if there are not enough forefront samples and include background samples is true 
                species_samples_num = len(sample_dict['forefront_sample_ids'])
                number_background_samples = sample_limit - species_samples_num #compute how many background samples are needed to meet limit
                #make sure remaining sample number does not exceed number of background samples
                number_background_samples = len(sample_dict['background_sample_ids']) if number_background_samples > len(sample_dict['background_sample_ids']) else number_background_samples
                sample_dict['background_sample_ids'] = sample_dict['background_sample_ids'][0:number_background_samples] #keep background samples
                species_samples_num += number_background_samples
            else: #else include all available forefront samples
                sample_dict['background_sample_ids'] = [] 
                species_samples_num = len(sample_dict['forefront_sample_ids'])
        else:
            species_samples_num = len(sample_dict['forefront_sample_ids']) + len(sample_dict['background_sample_ids'])

        print('Found {:>6} samples for species with key "{}"'.format(species_samples_num, species_key))

    #collect all samples into one list
    sample_ids_to_download = []
    for species_key, sample_dict in species_sample_ids.items():
        sample_ids_to_download += sample_dict['forefront_sample_ids']
        sample_ids_to_download += sample_dict['background_sample_ids']
    #remove duplicates from list
    sample_ids_to_download = set(sample_ids_to_download)
    #remove samples that have already been dowloaded
    sample_ids_to_download = list(sample_ids_to_download.difference(already_downloaded_sample_ids))

    return species_sample_ids, sample_ids_to_download
    
def download_by_sample_ids(sample_ids : list, samples_metadata : dict, destination_path):
    progress_bar = ProgressBar(len(sample_ids))
    for sample_id in sample_ids:
        file_path = os.path.join(destination_path, str(sample_id) + '.mp3')
        url_link = samples_metadata[sample_id]['recording_link']

        response = requests.get(url_link)
        if response: 
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print("ERROR CODE: {}! sample {}(gbif_id) with url {} could not be downloaded!".format(response.status_code, sample_id, url_link))
        
        progress_bar.progress()
        progress_bar.print()
        #count += 1
'''
def download_species_samples(species_sample_dict, destination_path, limit = float('inf')):
    samples_dict = species_sample_dict['samples']
    samples_path = os.path.join(destination_path, str(species_sample_dict['species_key']))

    sample_num = len(samples_dict) if limit == float('inf') else limit
    print("Downloading {} samples for {} with species key {}".format(sample_num, species_sample_dict['scientific_name'], species_sample_dict['species_key']))

    if not os.path.exists(samples_path):
        os.mkdir(samples_path)

    count = 0
    for gbif_id, sample_info in samples_dict.items():
        file_path = os.path.join(samples_path, str(gbif_id) + '.mp3')
        url_link = sample_info['recording_link']

        response = requests.get(url_link)
        if response: 
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print("ERROR CODE: {}! sample {}(gbif_id) with url {} could not be downloaded!".format(response.status_code, gbif_id, url_link))
        
        count += 1
        if (limit <= count):
            break
'''
def get_downloaded_sample_ids(download_dir):
    '''
    traverses the directory pointed to by download_dir and returns the ids of the downloaded samples
    '''
    #parse the directory and returns the file name as int (the file name should be the samples gbif id)
    return [int(os.path.splitext(f)[0]) for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]

def generate_download_species_sample_info(downloaded_sample_ids, samples_metadata_dict, download_species_sample_info_file_path):
    download_species_sample_info = {}
    for sample_id in downloaded_sample_ids:    
        species_key = samples_metadata_dict[sample_id]['forefront_bird_key']
        if not species_key in download_species_sample_info:
            download_species_sample_info[species_key] = {
                "forefront_sample_ids" : [sample_id],
                "background_sample_ids": []
            }
        else: 
            download_species_sample_info[species_key]["forefront_sample_ids"].append(sample_id)

    return download_species_sample_info
#horrible name
def add_species_samples_to_download_species_sample_info(download_species_sample_info, species_sample_ids):
    '''
    adds sample ids from the species_sample_ids dict to the download_species_sample_info dictionary. For both forefront_sample_ids
    and background_sample_ids it checks if the to be added samples already exists and if they do not it adds it to the relevant list of the
    species dict.
    '''
    for species_key, samples in species_sample_ids.items():
        if species_key in download_species_sample_info:
            for sample_id in samples["forefront_sample_ids"]:
                if sample_id not in download_species_sample_info[species_key]["forefront_sample_ids"]:
                    download_species_sample_info[species_key]["forefront_sample_ids"].append(sample_id)
            for sample_id in samples["background_sample_ids"]:
                if sample_id not in download_species_sample_info[species_key]["background_sample_ids"]:
                    download_species_sample_info[species_key]["background_sample_ids"].append(sample_id)
        else:
            download_species_sample_info[species_key] = species_sample_ids[species_key]

def add_sample_ids_to_download_species_sample_info(download_species_sample_info : dict, sample_ids : list, samples_metadata_dict : dict):
    '''
    adds sample ids in the sample_ids list to the download_species_sample_info dictionary. It uses the samples_metadata_dict to identify
    the main/forefront bird in the sample and appends the sample id to the forefront_sample_ids list using the species key of the bird.
    Furthermore it also adds the sample id to all background species keys which are noted in the sample's meta dictionary. 
    '''
    for sample_id in sample_ids: 
        species_key = samples_metadata_dict[sample_id]['forefront_bird_key'] #get forefront bird's species key  in sample 
        if not species_key in download_species_sample_info:                  #if the species is not in download_species_sample_info 
            download_species_sample_info[species_key] = {                    #add info dict of species with the sample_id in forefront_sample_ids
                "forefront_sample_ids" : [sample_id],
                "background_sample_ids": []
            }
        # if the species already exists in dictionary amd the sample is not in the samples
        elif sample_id not in download_species_sample_info[species_key]["forefront_sample_ids"]: 
            download_species_sample_info[species_key]["forefront_sample_ids"].append(sample_id)
        
        background_species_key = samples_metadata_dict[sample_id]['background_birds_keys']
        for species_key in background_species_key:
            if not species_key in download_species_sample_info:                  #if the species is not in download_species_sample_info 
                download_species_sample_info[species_key] = {                    #add info dict of species with the sample_id in forefront_sample_ids
                    "forefront_sample_ids" : [],
                    "background_sample_ids": [sample_id]
                }
            # if the species already exists in dictionary and the sample is not in the species's background samples
            elif sample_id not in download_species_sample_info[species_key]["background_sample_ids"]: 
                download_species_sample_info[species_key]["background_sample_ids"].append(sample_id)

    return download_species_sample_info


def empty_or_create_dir(dir_path):
    '''
    creates or empties the directory pointed to by dir_path
    '''
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {}. Exceptions: {}'.format(file_path, e)) 
    else:
        os.mkdir(dir_path)

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download mp3 samples from species described int the data_dictionary directory')
    parser.add_argument('--sample_min', type=int, required=False, default=0,
                            help =  'set the minimum forefront sample amount a species should have')
    parser.add_argument('--include_background_samples', required=False, action='store_true', 
                            help =  'if this argument is set, background samples are included in the \
                                    sample min argument and will be downloaded after all forefront samples (if download max has not been reached)')
    parser.add_argument('--download_max', type=int, required=False, default=None,
                            help =  'set the maximum samples that should be downloaded for each species')
    parser.add_argument('--exclude_unknown_species', required=False, action='store_true',
                            help =  'If this argument is set, unknown species, with species key 0, will be excluded from the download')
    parser.add_argument('--include_species_keys', type=int, required=False, default=[], nargs='+', 
                            help =  'Add keys of specific species which samples should be downloaded. \
                                    If this argument is given only samples for these species will be downloaded.')
    parser.add_argument('--include_sample_ids', type=int, required=False, default=[], nargs='+', 
                            help =  'Add ids of specific samples which should be downloaded. If this \
                                    argument is given only these samples will be downloaded.')
    parser.add_argument('--reset_download_dir', required=False, action='store_true',
                            help =  'If this argument is given, the **dataset/raw/** directory will be \
                                    completely emptied before samples are downloaded')
    args = parser.parse_args()

    #load data dictionary
    data_dictionary_path = os.path.join(dataset_path, 'data_dictionary')
    species_info_path = os.path.join(data_dictionary_path, 'species_info.csv')
    species_sample_info_file_path = os.path.join(data_dictionary_path, 'species_sample_info.json')
    samples_metadata_file_path = os.path.join(data_dictionary_path, 'samples_metadata.json')

    download_species_sample_info_file_path = os.path.join(data_dictionary_path, 'download_species_sample_info.json')


    species_info_df = pd.read_csv(species_info_path)
    #load species sample info
    with open(species_sample_info_file_path) as f:
        species_sample_info_dict = {int(species_key): info for species_key, info in json.load(f).items()} 
    #load samples metadata
    with open(samples_metadata_file_path) as f:
        samples_metadata_dict = {int(species_key): info for species_key, info in json.load(f).items()} #json.load(f)

    #check command line arguments

    raw_data_path = os.path.join(dataset_path, 'raw')
    #if reset_download_dir is given delete all samples in dir
    if not os.path.isdir(raw_data_path): #check if raw direcotry exists in dataset/
        print("Creating '{}' directory for downloads.".format(raw_data_path))
        empty_or_create_dir(raw_data_path)
        downloaded_sample_ids = []          
        download_species_sample_info = {}
    else:
        if args.reset_download_dir:
            print("Reseting '{}' dir".format(raw_data_path))
            empty_or_create_dir(raw_data_path)
            downloaded_sample_ids = []
            download_species_sample_info = {}
        else:   #load already downloaded samples and load or create download_species_sample_info
            downloaded_sample_ids = get_downloaded_sample_ids(raw_data_path) #get ids of samples that have already been downloaded
            if not os.path.exists(download_species_sample_info_file_path):
                download_species_sample_info = generate_download_species_sample_info(downloaded_sample_ids, samples_metadata_dict, download_species_sample_info_file_path)
            else: 
                with open(download_species_sample_info_file_path) as f:
                    download_species_sample_info = {int(species_key): info for species_key, info in json.load(f).items()}
    #make sure only one of the two options is given
    if args.include_species_keys and args.include_sample_ids:
        raise Exception('Both include_species_keys and include_sample_ids were given! Please only give one or the other!')

    if args.include_species_keys:
        #check if all given keys are in the dictionary
        species_key_list = species_info_df['species_key'].tolist()
        keys_not_found_arr = [key for key in args.include_species_keys if key not in species_key_list]

        #print(species_info_df[species_info_df['species_key'] == 2473663])
        if keys_not_found_arr: 
            print('WARNING! Some given species keys were not found in the data dictionary: {}'.format(keys_not_found_arr))
            bool_input = input('Would you like to continue? (y/n)')[0].lower()
            if bool_input == 'y':
                pass
            else:
                print("Quiting script")
                quit()
        # remove all species rows which are not in args.include_species_keys
        species_indices = species_info_df[~species_info_df['species_key'].isin(args.include_species_keys)].index
        species_info_df.drop(species_indices, inplace = True)

    elif args.include_sample_ids:
        pass
    else:
        #exclude_unknown_species is given drop species with specie_key = 0
        if args.exclude_unknown_species:
            unknown_species_index = species_info_df[species_info_df['species_key'] == 0].index
            species_info_df.drop(unknown_species_index, inplace = True)
        #if include_background_samples is given include background samples in sample minimum
        if args.include_background_samples:
            not_meeting_requirements_indices = species_info_df[species_info_df['forefront_recordings'] + species_info_df['background_recordings'] < args.sample_min].index
        else:
            not_meeting_requirements_indices = species_info_df[species_info_df['forefront_recordings'] < args.sample_min].index
        species_info_df.drop(not_meeting_requirements_indices, inplace = True)

        print("{} species have more than {} samples".format(len(species_info_df), args.sample_min))

    if not args.include_sample_ids:
        species_keys = species_info_df['species_key'].tolist()
        if not species_keys:
            print("No species met the given requirements. Ending script")
            quit()
        #filter samples info to only relevant samples
        species_sample_data = get_species_sample_ids(species_sample_info_dict, species_keys)
        species_sample_ids, sample_ids_to_download = filter_species_samples(species_sample_data, args.include_background_samples, args.download_max, downloaded_sample_ids)
        add_species_samples_to_download_species_sample_info(download_species_sample_info, species_sample_ids)
    else:
        #check if all given sample ids are in the sample metadata dictionary
        sample_ids_not_found_arr = [s_id for s_id in args.include_sample_ids if s_id not in samples_metadata_dict]
        if sample_ids_not_found_arr: 
            print('WARNING! Some given sample ids were not found in the data dictionary: {}'.format(sample_ids_not_found_arr))
            bool_input = input('Would you like to continue? (y/n)')[0].lower()
            if bool_input == 'y':
                pass
            else:
                print("Quiting script")
                quit()

        found_species_sample_ids = [s_id for s_id in args.include_sample_ids if s_id in samples_metadata_dict]

        sample_ids_to_download = set(found_species_sample_ids)
        #remove samples that have already been dowloaded
        sample_ids_to_download = list(sample_ids_to_download.difference(downloaded_sample_ids))
        add_sample_ids_to_download_species_sample_info(download_species_sample_info, sample_ids_to_download, samples_metadata_dict)

    print("A total of {} new samples will be downloaded".format(len(sample_ids_to_download)))

    download_by_sample_ids(sample_ids_to_download, samples_metadata_dict, raw_data_path)
    print()
    print("Completed downloading all samples")

    with open(download_species_sample_info_file_path, 'w') as outfile:
        json.dump(download_species_sample_info, outfile, indent=4)
