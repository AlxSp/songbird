
from typing import List, Dict
from enum import Enum
import argparse
import datetime
import os
import json

this_file_dir = os.path.dirname(os.path.abspath(__file__)) #absolute path to this file's directory
project_base_dir = os.path.dirname(os.path.dirname(this_file_dir)) #path to base dir of project
data_dir = os.path.join(project_base_dir, 'data')  #path to data_dir

class SampleRecordingType(Enum):
    """
    Defines the different of type of sample recordings.
    """
    Foreground = 'forefront_sample_ids'
    Background = 'background_sample_ids'


class DatasetInfo:
    """
    Class for querying information of the dataset.
    """
    _dataset_info_dir = 'data_dictionary'
    _downloaded_samples_file = 'download_species_sample_info.json'              # this file holds the species_id and sample_id for each sample that has been downloaded
    _failed_to_download_sample_ids_file = 'failed_to_download_sample_ids.txt'   # this file holds the sample ids for the samples that failed to download
    _samples_metadata_file = 'samples_metadata.json'                            # this file holds the metadata for each downloaded sample
    _species_info_file = 'species_info.csv'                                     # this file holds general information about the species                                 
    _species_samples_info_file = 'species_samples_info.json'                    # this file holds the relation between species_ids and sample_ids

    def __init__(self, dataset_info_dir = None) -> None:
        self.dataset_info_dir = os.path.join(data_dir, self._dataset_info_dir) if dataset_info_dir is None else dataset_info_dir

        with open(os.path.join(self.dataset_info_dir, self._samples_metadata_file)) as f:
            self.samples_metadata = json.load(f)
            
        self.downloaded_species_samples_dict = self._create_downloaded_species_samples_dict(os.path.join(data_dir, 'raw'), self.samples_metadata)
        
        with open(os.path.join(self.dataset_info_dir, self._failed_to_download_sample_ids_file)) as f:
            self.failed_to_download_sample_ids = f.read().split(',')
        with open(os.path.join(self.dataset_info_dir, self._species_info_file)) as f:
            self.species_info = f.read().split('\n')

    def _create_downloaded_species_samples_dict(self, raw_samples_dir: str, samples_metadata_dict) -> Dict:
        """
        Load the downloaded samples file and also check the raw directory for files that are not in the downloaded file.
        :param downloaded_samples_file: downloaded samples file
        :return: downloaded samples
        """
        downloaded_species_samples_dict = {}
        
        downloaded_sample_id_arr = [file_id.split('.')[0] for file_id in os.listdir(raw_samples_dir)]
        
        # print(f"Found {len(downloaded_sample_id_arr)} downloaded samples.")
            
        for sample_id in downloaded_sample_id_arr:
            foreground_species_key = samples_metadata_dict[sample_id]['forefront_bird_key']
            if not foreground_species_key in downloaded_species_samples_dict:
                downloaded_species_samples_dict[foreground_species_key] = {
                    SampleRecordingType.Foreground.value: [sample_id], 
                    SampleRecordingType.Background.value: []
                }
            else:
                downloaded_species_samples_dict[foreground_species_key][SampleRecordingType.Foreground.value].append(sample_id)
                
            background_species_key_arr = samples_metadata_dict[sample_id]['background_birds_keys']
            
            for background_species_key in background_species_key_arr:
                if not background_species_key in downloaded_species_samples_dict:
                    downloaded_species_samples_dict[background_species_key] = {
                        SampleRecordingType.Foreground.value: [], 
                        SampleRecordingType.Background.value: [sample_id]
                    }
                else:
                    downloaded_species_samples_dict[background_species_key][SampleRecordingType.Background.value].append(sample_id)
                    
        return downloaded_species_samples_dict
     
    
     
    def describe_downloaded_samples(self, show_top_n = 10) -> None:
        """
        Prints a description of the downloaded samples.
        :return: None
        """
        species_with_foreground_samples = []
        species_with_background_samples = []
        
        for species_key in self.downloaded_species_samples_dict:
            foreground_samples = self.downloaded_species_samples_dict[species_key][SampleRecordingType.Foreground.value]
            background_samples = self.downloaded_species_samples_dict[species_key][SampleRecordingType.Background.value]
            
            number_of_foreground_samples = len(foreground_samples)
            number_of_background_samples = len(background_samples)
            
            if number_of_foreground_samples > 0:
                samples_length_sec = []
                for sample_id in foreground_samples:
                    recoding_time_sec = self.samples_metadata[sample_id].get('recording_time_sec', None)
                    recoding_time_sec = recoding_time_sec if recoding_time_sec is not None else 0 
                    samples_length_sec.append(recoding_time_sec)
                total_samples_length_sec = sum(samples_length_sec)
                species_with_foreground_samples.append((species_key, number_of_foreground_samples, total_samples_length_sec))
            
            if number_of_background_samples > 0:
                #sample_length_sec = sum([self.samples_metadata[sample_id].get('recording_time_sec', 0) for sample_id in background_samples])
                species_with_background_samples.append((species_key, number_of_background_samples, 0))
            
        species_with_foreground_samples.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Samples for {len(species_with_foreground_samples)} species with foreground recordings")
        print(f"The top {show_top_n} species with foreground recordings:")
        for i in range(show_top_n):
            print(f"Species key: {species_with_foreground_samples[i][0]} Number of samples: {species_with_foreground_samples[i][1]} Total length: {str(datetime.timedelta(seconds=species_with_foreground_samples[i][2]))}") #
            
    def get_downloaded_species_sample_ids(self, species_key: int, sample_recording_type: SampleRecordingType = None) -> list:
        """
        Get a list of sample ids for a given species.
        :param species_id: species id
        :param sample_recording_type: type of sample recording
        :return: list of sample ids
        """
        
        try:
            if sample_recording_type is None:
                print(f"Species key: {species_key} - Downloaded samples: {len(self.downloaded_species_samples_dict[species_key][SampleRecordingType.Foreground.value]) + len(self.downloaded_species_samples_dict[species_key][SampleRecordingType.Background.value])}")
                return self.downloaded_species_samples_dict[species_key][SampleRecordingType.Foreground.value] + self.downloaded_species_samples_dict[species_key][SampleRecordingType.Background.value]
            else:
                print(f"Species key: {species_key} - {sample_recording_type.value} - Downloaded samples: {len(self.downloaded_species_samples_dict[species_key][sample_recording_type.value])}")
                return self.downloaded_species_samples_dict[species_key][sample_recording_type.value]

        except Exception as e:
            print(f"Error retrieving sample ids! Error: {e}")
            
    def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get the metadata of a given file.
        :param file_id: file id
        :return: metadata of file
        """
        return self.samples_metadata[file_id]
        


if __name__ == "__main__":
    dataset_info = DatasetInfo()
    #print(dataset_info.get_downloaded_species_sample_ids(1))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dbs','--describe_downloaded_samples', type = int, const = 10, nargs = '?', default = 0, help = 'Return list of downloaded samples')
    args = parser.parse_args()
    
    if args.describe_downloaded_samples:
        dataset_info.describe_downloaded_samples(args.describe_downloaded_samples)
    
    
    