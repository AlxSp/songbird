
from typing import List, Dict
from enum import Enum
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

        with open(os.path.join(self.dataset_info_dir, self._downloaded_samples_file)) as f:
            self.downloaded_samples = json.load(f)
        with open(os.path.join(self.dataset_info_dir, self._failed_to_download_sample_ids_file)) as f:
            self.failed_to_download_sample_ids = f.read().split(',')
        with open(os.path.join(self.dataset_info_dir, self._samples_metadata_file)) as f:
            self.samples_metadata = json.load(f)
        with open(os.path.join(self.dataset_info_dir, self._species_info_file)) as f:
            self.species_info = f.read().split('\n')


    def get_download_sample_ids(self, species_id: int, sample_recording_type: SampleRecordingType = None) -> list:
        """
        Get a list of sample ids for a given species.
        :param species_id: species id
        :param sample_recording_type: type of sample recording
        :return: list of sample ids
        """
        
        try:
            species_id = str(species_id)
            if sample_recording_type is None:
                return self.downloaded_samples[species_id][SampleRecordingType.Foreground.value] + self.downloaded_samples[species_id][SampleRecordingType.Background.value]
            else:
                return self.downloaded_samples[species_id][sample_recording_type.value]

        except Exception as e:
            print(f"Error retrieving sample ids! Error: {e}")


if __name__ == "__main__":
    dataset_info = DatasetInfo()
    #print(dataset_info.get_download_sample_ids(1))