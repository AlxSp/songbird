import pandas as pd
import json
import datetime
import time
import argparse
import os
import requests
import shutil
import sys

this_file_dir = os.path.dirname(os.path.abspath(__file__)) #absolute path to this file's directory
project_base_dir = os.path.dirname(os.path.dirname(this_file_dir)) #path to base dir of project
data_dir = os.path.join(project_base_dir, 'data')  #path to data_dir

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
    
def download_by_sample_ids(sample_ids : list, samples_metadata : dict, destination_path, download_delay_sec = 0.0):
    progress_bar = ProgressBar(len(sample_ids))
    
    failed_to_download_sample_ids = []

    for sample_id in sample_ids:
        file_path = os.path.join(destination_path, str(sample_id) + '.mp3')
        url_link = samples_metadata[sample_id]['recording_link']

        download_error = False
        try:
            response = requests.get(url_link)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            add_sample_id_to_download_species_sample_info(download_species_sample_info, sample_id, samples_metadata)
        except requests.exceptions.Timeout as e:
            print(f"TIME OUT ERROR for sample {sample_id}(gbif_id) with url {url_link}! {e}")
            download_error = True
        except requests.exceptions.TooManyRedirects as e:
            print(f"TOO MANY REDIRECTS ERROR for sample {sample_id}(gbif_id) with url {url_link}! {e}")
            download_error = True
        except requests.exceptions.RequestException as e:
            print(f"REQUEST EXCPETION for sample {sample_id}(gbif_id) with url {url_link}! {e}")
            download_error = True
        except requests.exceptions.HTTPError as e:
            print(f"URL ERROR for sample {sample_id}(gbif_id) with url {url_link}! {e}")
            download_error = True

        if download_error:
            failed_to_download_sample_ids.append(sample_id)
        
        time.sleep(download_delay_sec)
        progress_bar.progress()
        progress_bar.print()

    return failed_to_download_sample_ids

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

def add_sample_id_to_download_species_sample_info(download_species_sample_info : dict, sample_id : int, samples_metadata_dict : dict):
    '''
    adds sample id to the download_species_sample_info dictionary. It uses the samples_metadata_dict to identify
    the main/forefront bird in the sample and appends the sample id to the forefront_sample_ids list using the species key of the bird.
    Furthermore it also adds the sample id to all background species keys which are noted in the sample's meta dictionary. 
    '''
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

def add_sample_ids_to_download_species_sample_info(download_species_sample_info : dict, sample_ids : list, samples_metadata_dict : dict):
    '''
    adds sample ids in the sample_ids list to the download_species_sample_info dictionary. It uses the samples_metadata_dict to identify
    the main/forefront bird in the sample and appends the sample id to the forefront_sample_ids list using the species key of the bird.
    Furthermore it also adds the sample id to all background species keys which are noted in the sample's meta dictionary. 
    '''
    for sample_id in sample_ids: 
        add_sample_id_to_download_species_sample_info(download_species_sample_info, sample_id, samples_metadata_dict)

def get_sample_ids_from_txt(file_path):
    sample_ids  = []    # empty sample ids array
    with open(file_path) as f:  # open file
        text_file_data = f.readlines()  #read all lines in file
        for line in text_file_data: # for each line
            split_values = line.replace(" ","").rstrip('\n').split(',') # remove whitespaces, new line characters and split line by commas into separate sample ids
            sample_ids += split_values  # append split samples ids

    return sample_ids

def get_sample_ids_from_csv(file_path):
    return pd.read_csv(file_path)['sample_id'].to_list() # read csv file with pandas (expecting a header)

def get_sample_ids_from_downloaded_samples_json(file_path):
    sample_ids  = []    # empty sample ids array
    with open(file_path) as f:
        downloaded_samples = json.load(f) #load samples dictionary
    species_samples_list = [samples["forefront_sample_ids"] + samples["background_sample_ids"] for key, samples in downloaded_samples.items()]
    for species_samples in species_samples_list:
        sample_ids += species_samples
    return sample_ids

def use_sample_ids_from_file(file_path):
    if not os.path.isfile(file_path):
        raise Exception(f'file at given path "{file_path}" does not exist!')

    string_sample_ids = []
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == ".txt":
        string_sample_ids = get_sample_ids_from_txt(file_path)
    elif file_extension == ".csv":
        string_sample_ids = get_sample_ids_from_csv(file_path)
    elif file_extension == ".json":
        string_sample_ids = get_sample_ids_from_downloaded_samples_json(file_path)

    return [int(sample_id) for sample_id in string_sample_ids]

def filter_sample_ids_by_length(sample_ids, max_sample_length, samples_metadata_dict):
    """
    Filters the given samples if they are equal or below the max_sample_length parameter
    """
    before_len = len(sample_ids)
    filtered_sample_ids = [sample_id for sample_id in sample_ids if samples_metadata_dict[sample_id].get('recording_link', sys.maxsize) <= max_sample_length]
    after_len = len(filtered_sample_ids)
    print(f"Filtered {before_len - after_len} samples that exceed the max_sample_length; {max_sample_length} second")
    return filtered_sample_ids

def compile_sample_id_list_from_args(args, downloaded_sample_ids, samples_metadata_dict):
    #create paths to meta and info file    
    sample_ids_to_download = []

    if (args.use_species_keys and args.use_sample_ids) or (args.all and args.use_species_keys) or (args.all and args.use_sample_ids):   #make sure only one of the two options is given
        raise Exception('Please choose only one of use_species_keys, use_sample_ids, or all! For more info execute the script with the -h command.')

    if args.all or args.use_species_keys:

        species_info_df = pd.read_csv(species_info_path)

        if args.all:
            print("Gathering sample information for all species!")
            #exclude_unknown_species is given drop species with specie_key = 0
            if args.exclude_unknown_species:
                unknown_species_index = species_info_df[species_info_df['species_key'] == 0].index
                species_info_df.drop(unknown_species_index, inplace = True)
            #if include_background_samples is given include background samples in sample minimum
            if args.include_background_samples:
                not_meeting_requirements_indices = species_info_df[species_info_df['forefront_recordings'] + species_info_df['background_recordings'] < args.species_sample_min].index
            else:
                not_meeting_requirements_indices = species_info_df[species_info_df['forefront_recordings'] < args.species_sample_min].index

            species_info_df.drop(not_meeting_requirements_indices, inplace = True)

            print("{} species have more than {} samples".format(len(species_info_df), args.species_sample_min))

        elif args.use_species_keys:
            species_key_list = species_info_df['species_key'].tolist()
            keys_not_found_arr = [key for key in args.use_species_keys if key not in species_key_list]
            #if some keys were not found; print them out and ask if the user wishes to continue
            if keys_not_found_arr: 
                print('WARNING! Some given species keys were not found in the data dictionary: {}'.format(keys_not_found_arr))
                bool_input = input('Would you like to continue with the known species keys? (y/n)')[0].lower()
                if bool_input == 'y':
                    pass
                else:
                    print("Quiting script")
                    quit()

            # remove all species rows which are not in args.use_species_keys
            species_indices = species_info_df[~species_info_df['species_key'].isin(args.use_species_keys)].index
            species_info_df.drop(species_indices, inplace = True)

        # create list of species keys that fit user requirements
        species_keys = species_info_df['species_key'].tolist()
        if not species_keys:    #check if the list is not empty
            print("No species met the given requirements. Ending script")
            quit()
        
        print(f"Gathering sample information for {len(species_keys)} species!")

        species_sample_info_file_path = os.path.join(data_dictionary_path, 'species_sample_info.json')
        #load species sample info, which describes in what samples the species appear 
        with open(species_sample_info_file_path) as f:
            species_sample_info_dict = {int(species_key): info for species_key, info in json.load(f).items()} 
        #filter samples info to only relevant samples
        species_sample_data = get_species_sample_ids(species_sample_info_dict, species_keys)
        species_sample_ids, sample_ids_to_download = filter_species_samples(species_sample_data, args.include_background_samples, args.species_sample_max, downloaded_sample_ids)
        #add_species_samples_to_download_species_sample_info(download_species_sample_info, species_sample_ids)
    
    elif args.use_sample_ids_from_file or args.use_sample_ids:
        if args.use_sample_ids_from_file:   #if the user provided a file, get the samples ids from the file
            input_sample_ids = use_sample_ids_from_file(args.use_sample_ids_from_file)
            print(f"Gathering {len(input_sample_ids)} sample ids from command line input!")
        elif args.use_sample_ids: # else get the sample ids as a list from the user argument
            input_sample_ids = args.use_sample_ids
            print(f"Gathering {len(input_sample_ids)} sample ids from file!")

        sample_ids_not_found_arr = [s_id for s_id in input_sample_ids if s_id not in samples_metadata_dict]
        if sample_ids_not_found_arr: 
            print('WARNING! Some given sample ids were not found in the data dictionary: {}'.format(sample_ids_not_found_arr))
            bool_input = input('Would you like to continue with the known sample ids? (y/n)')[0].lower()
            if bool_input == 'y':
                pass
            else:
                print("Quiting script")
                quit()

        sample_ids_to_download = [s_id for s_id in input_sample_ids if s_id in samples_metadata_dict]

    else:
        raise Exception('Please choose only one of use_species_keys, use_sample_ids, or all! For more info execute the script with the -h command.')
    sample_ids_to_download = set(sample_ids_to_download)
    print(f"Found {len(sample_ids_to_download)} samples to download")
    #remove samples that have already been dowloaded

    sample_ids_to_download = list(sample_ids_to_download.difference(downloaded_sample_ids))
    print(f"Found {len(downloaded_sample_ids)} samples that have already been download")
    if args.sample_length_max is not None:
        sample_ids_to_download = filter_sample_ids_by_length(sample_ids_to_download, args.sample_length_max, samples_metadata_dict)

    return sample_ids_to_download

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download mp3 samples from species described int the data_dictionary directory')
    parser.add_argument('--all', required=False, action='store_true', 
                            help =  'if this argument is set, all species and their samples will be downloaded')

    parser.add_argument('--exclude_unknown_species', required=False, action='store_true',
                            help =  'If this argument is set, unknown species, with species key 0, will be excluded from being download. \
                                    This argument is only valid when the --all argument is given')
    parser.add_argument('--species_sample_min', type=int, required=False, default=0,
                            help =  'set the minimum forefront sample amount a species should have. \
                                    This argument is only valid when the --all argument is given')

    parser.add_argument('--use_species_keys', type=int, required=False, default=[], nargs='+', 
                            help =  'Add keys of specific species which samples should be downloaded. \
                                    If this argument is given only samples for these species will be downloaded.')

    parser.add_argument('--include_background_samples', required=False, action='store_true', 
                            help =  'if this argument is set, background samples are included in the \
                                    species samples and sample min argument and will be downloaded after all forefront samples (if download max has not been reached) \
                                    This argument is only valid when the --all or --use_species_keys argument is given')
    parser.add_argument('--species_sample_max', type=int, required=False, default=None,
                            help =  'set the maximum samples that should be downloaded for each species \
                                    This argument is only valid when the --all or --use_species_keys argument is given')

    parser.add_argument('--use_sample_ids', type=int, required=False, default=[], nargs='+', 
                            help =  'Add ids of specific samples which should be downloaded. If this \
                                    argument is given only these samples will be downloaded.')
    parser.add_argument('--use_sample_ids_from_file', type=str, required=False, 
                            help =  'Give a path to a txt, download_species_sample_info.json or csv file which specifies sample ids which should be downloaded. \
                                     If a csv file is given the header of sample id column has to be named "sample_id"')

    parser.add_argument('--sample_length_max', type=int, required=False, default=None,
                            help =  'set the maximum sample length in sec that a sample should have to be downloaded \
                                    This argument is valid with --all, --use_species_keys, --use_sample_ids, and use --use_sample_ids_from_file argument is given')               
    
    parser.add_argument('--reset_download_dir', required=False, action='store_true',
                            help =  'If this argument is given, the **data/raw/** directory will be \
                                    completely emptied before samples are downloaded. And the download_species_sample_info.json file will be removed')
    
    parser.add_argument('--download_delay', type=float, required=False, default=0,
                            help =  'Set a delay in seconds between downloads to minimize server load')

    args = parser.parse_args()

    raw_data_path = os.path.join(data_dir, 'raw') # global variable that holds the path to the raw downloads directory
    data_dictionary_path = os.path.join(data_dir, 'data_dictionary')
    species_info_path = os.path.join(data_dictionary_path, 'species_info.csv')
    samples_metadata_file_path = os.path.join(data_dictionary_path, 'samples_metadata.json')
    download_species_sample_info_file_path = os.path.join(data_dictionary_path, 'download_species_sample_info.json')
    failed_to_download_sample_ids_file_path = os.path.join(data_dictionary_path, 'failed_to_download_sample_ids.txt')

    #load samples metadata, which holds general information on each sample
    with open(samples_metadata_file_path) as f:
        samples_metadata_dict = {int(species_key): info for species_key, info in json.load(f).items()}

    downloaded_sample_ids = []          # list of downloaded sample ids
    download_species_sample_info = {}   # dictionary of downloaded species and their samples

    if os.path.isdir(raw_data_path): #if data/raw does exist
        if args.reset_download_dir:  #if reset download dir is given, delete all samples in dir
            print("Reseting '{}' dir".format(raw_data_path))
            empty_or_create_dir(raw_data_path)
            try:    # try to remove the files if they exist
                os.remove(download_species_sample_info_file_path)
                os.remove(failed_to_download_sample_ids_file_path)
            except:
                pass 
        else:   #load already downloaded samples and load or create download_species_sample_info
            downloaded_sample_ids = get_downloaded_sample_ids(raw_data_path) #get ids of samples that have already been downloaded
            if not os.path.exists(download_species_sample_info_file_path):  # if download sample info file does not exist
                #create the downloaded sample dictionary from the samples in the raw directory
                download_species_sample_info = generate_download_species_sample_info(downloaded_sample_ids, samples_metadata_dict, download_species_sample_info_file_path)
            else:   # create a dictionary from the download species sample info
                with open(download_species_sample_info_file_path) as f:
                    download_species_sample_info = {int(species_key): info for species_key, info in json.load(f).items()}
    else:   # if data does not exist, create it
        print("Creating '{}' directory for downloads.".format(raw_data_path))
        empty_or_create_dir(raw_data_path)

    sample_ids_to_download = compile_sample_id_list_from_args(args, downloaded_sample_ids, samples_metadata_dict)  #create a list of sample ids to be downloaded from user input

    print("A total of {} new samples will be downloaded".format(len(sample_ids_to_download)))

    failed_to_download_sample_ids = download_by_sample_ids(sample_ids_to_download, samples_metadata_dict, raw_data_path, args.download_delay) #down load sample ids
    print()
    print("Completed downloading all samples")

    with open(download_species_sample_info_file_path, 'w') as outfile: #create download species sample info 
        json.dump(download_species_sample_info, outfile, indent=4)

    if failed_to_download_sample_ids: #if samples failed to download; list them in file
        write_mode = 'a' if os.path.exists(failed_to_download_sample_ids_file_path) else 'w' # if file already exists; append to it, otherwise create it 
        with open(failed_to_download_sample_ids_file_path, write_mode) as outfile:
            for sample_id in failed_to_download_sample_ids:
                outfile.write(f"{sample_id}, ")
