# Songbird Project


## Requirements 
* Python 3.6+

### Modules:
* requests
* librosa
* matplotlib
* numpy
* pandas

## Setup: 
1. Download the dataset: [Xeno-canto - Bird sounds from around the world](https://www.gbif.org/dataset/b1047888-ae52-4179-9dd5-5448ea342a24#methodology)
1. Extract it into the xeno_canto_bsfatw directory
1. run create_data_dictionary.py
    - to check out info on the dataset run the meta_data_analysis.ipynb notebook
1. run download_data.py (Currently downloads only 1 sample)
1. run audio_analysis for info on the sample

---

## Scripts

* ### create_data_dictionary.py
    Processes files and information in the **xeno_canto_bsfatw/** directory (Setup steps 1 & 2) and creates a data dictionary in the **dataset/** directory. The created files are; 
    species_info.csv, species_sample_info.json and samples_metadata.json and are located in the **dataset/data_dictionary** directory. These files are utilized by the `download_data.py` script.

* ### download_data.py

    Reads the data dictionary (**dataset/data_dictionary/**) and downloads files in the mp3 format to the **dataset/raw/** directory. It will not download a sample if it is already present in the **dataset/raw/** directory. Creates a download_species_sample_info.json file in the **dataset/data_dictionary/** directory which stores information on downloaded samples and their relation to the species

    **Arguments:**
    
    * **all**

        set if all available samples should be downloaded 

    * **exclude_unknown_species**

        If this argument is set, unknown species, with species key 0, will be excluded from the download. This argument is only valid when the --all argument is given

    * **species_sample_min <int_value>**

        set the minimum forefront sample amount a species should have. This argument is only valid when the --all argument is given

    * **use_species_keys <int_value 0> ... <int_value n>**

        Add keys of specific species which samples should be downloaded. If this argument is given only samples for these species will be downloaded.

    * **include_background_samples**

        If this argument is given, background samples are included in the sample min argument and will be downloaded after all forefront samples (if download max has not been reached). This argument is only valid when the --all or --use_species_keys argument is given

    * **species_sample_max <int_value>**

        set the maximum samples that should be downloaded for each species. This argument is only valid when the --all or --use_species_keys argument is given

    * **use_sample_ids <int_value 0> ... <int_value n>**

        Add ids of specific samples which should be downloaded. If this argument is given only these samples will be downloaded.

    * **use_sample_ids_from_file <int_value 0> ... <int_value n>**

        Give a path to a txt, download_species_sample_info.json or csv file which specifies the sample ids which should be downloaded. If a csv file is given the header of the sample id column has to be named "sample_id"

    * **sample_length_max <int_value>**

        set the maximum sample length in sec that a sample should have to be downloaded
        This argument is valid with --all, --use_species_keys, --use_sample_ids, and use --use_sample_ids_from_file argument is given

    * **reset_download_dir**

        If this argument is given, the **dataset/raw/** directory will be completely emptied before samples are downloaded
    
## Jupyter Notebooks

* ### audio_sample_analysis.ipynb

    Use this notebook to quickly checkout the wave plot and mel spectogram of downloaded samples.


## Project Structure:
```
├──xeno_canto_bsfatw  
│   ├── occurrence.txt  
│   ├── ...
│   └── multimedia.txt
│
├──dataset  
│   ├──data_dictionary
│   │   ├── species_info.csv
│   │   ├── species_sample_info.json
│   │   └── samples_metadata.json
│   ├──raw
│   └──processed
├── scripts    
└── jupyter notebooks
```

Dictionary in samples_metadata.json
```python
sample_dict = {
    'gbifID' : int, #unique gbifID 
    'recording_link' : string, #link to media
    'decimal_latitude' : float,
    'decimal_longitude': float,
    'date' : datetime_string,
    'behavior' : [], # > array of strings 
    'forefront' : int,
    'background_birds' : [], # > array of species key ints 
}
```
---

## References:

### Audio Analysis:
* [Mel Frequency Cepstral Coefficient (MFCC) tutorial by practical cryptography](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
* [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
* [Topic: Spectrogram, Cepstrum and Mel-Frequency Analysis](http://www.speech.cs.cmu.edu/15-492/slides/03_mfcc.pdf)

### Related Research Papers:
* to be added

### Related Projects:
* [Acoustic Detection of Humpback Whales Using a Convolutional Neural Network](https://ai.googleblog.com/2018/10/acoustic-detection-of-humpback-whales.html)

### ML Repository Structure:
* [ReproduciblePython](https://github.com/trallard/ReproduciblePython)
* [Structure and automated workflow for a machine learning project — part 1](https://towardsdatascience.com/structure-and-automated-workflow-for-a-machine-learning-project-2fa30d661c1e)
* [TensorFlow: A proposal of good practices for files, folders and models architecture](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3)
* [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science)
---
current main sample gbif_id:

2243804495








Possbile Main Candidates:  
red crossbill,loxia curvirostra,9629160