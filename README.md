# Songbird Project


## Python Version Requirement 
* Python 3.8+

## Setup: 

### Creating the environment 
1. Create conda enviroment: 
    ```
    conda create songbird --file requirements.txt -c conda-forge
    ```

    or 

    ```
    conda env create -f environment.yml
    ```


### Accessing and processing the data

1. Download the dataset: [Xeno-canto - Bird sounds from around the world](https://www.gbif.org/dataset/b1047888-ae52-4179-9dd5-5448ea342a24#methodology)
1. Extract it into the `data/xeno_canto_bsfatw` directory
1. run create_data_dictionary.py
    - to check out info on the dataset run the meta_data_analysis.ipynb notebook
1. run download_data.py (Currently downloads only 1 sample)
1. run audio_analysis for info on the sample

---

## Scripts in `songbird/`

### data/

* ### create_data_dictionary.py
    Processes files and information in the **xeno_canto_bsfatw/** directory (Setup steps 1 & 2) and creates a data dictionary in the **dataset/** directory. The created files are; 
    species_info.csv, species_sample_info.json and samples_metadata.json and are located in the **dataset/data_dictionary** directory. These files are utilized by the `download_data.py` script.

* ### download_data.py

    Reads the data dictionary (**dataset/data_dictionary/**) and downloads files in the mp3 format to the **dataset/raw/** directory. It will not download a sample if it is already present in the **dataset/raw/** directory. Creates a download_species_sample_info.json file in the **dataset/data_dictionary/** directory which stores information on downloaded samples and their relation to the species

    **CLI Arguments:**
    
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

    * **download_delay <float_value>**

        Include this argument with a float representing the seconds of delay between each download
----

### audio/

* ### generate_audio_events.py

    Processes audio files and extracts the detected audio events in csv format. Also allows generation of plots for debugging.

    **CLI Arguments:**
    
    * **all_samples**

        set if all available samples should be processed for audio event detection.  

    * **sample_ids <int_value 0> ... <int_value n>**

        Add ids of specific samples which should be processed.    
    
    * **generate_process_plots**

        set if plots of audio detection should be created (this feature is mainly useful for debugging). Plots will be stored in the **audio_event_plots/** directory.

    * **multi_processing <int_value (optional)>**

        set to use multiple cores for the processing of audio files. The parameter expects a following integer value specifying the number of cores that should be used; if only **--multi_processing** is given all available cores will be used.

* ### audio_processing.py

    holds the function for audio processing as well as pipelines. It's main purpose is to be imported and modified by another script, but it can be called with the same command line interface arguments as `generate_audio_events.py`. 

    Main functions & arguments:

        detect_audio_events_with_zscore(
        sample_id
        audio_conversion_parameters, 
        event_detection_parameters, 
        clustering_parameters, 
        event_processing_parameters, 
        additional_parameters 
        )

    Custom function: 

        detect_audio_events_with_custom(
        sample_id, 
        audio_conversion_parameters, 
        audio_processing_parameters,
        event_detection_parameters, 
        clustering_parameters, 
        event_processing_parameters, 
        additional_parameters 
        )

    The main functions follow a format which makes it easy to adjust parameters or swap out entire functions:

    audio_conversion: parameters to convert the audio from file to array and to a spectrogram
    audio_processing: parameters to alter the spectrogram
    event_detection:  parameters to detect audio anomalies or peaks in the spectrogram
    clustering: parameters to cluster the peaks into events
    event_processing: parameters to convert the events into a standard format of start_time, end_time, max_frequency and min_frequency

    ### Custom function pipeline:

    #### 1. audio processing function (`fn_process_spectrogram(spectrogram)`):
        
    > The purpose of this function is to further manipulate the spectrogram before peaks are detected.
        
    Input:    
    - a 2d array  which represents a spectrogram. Input format is ( time_steps, frequency_bins ).
    
    Output: 
    - a 2d array  which represents a spectrogram. Output format is ( time_steps, frequency_bins ). 

    #### 2. peak detection function (`fn_detect_peaks_in_spectrogram(transposed_spectrogram, event_detection_parameters)`):
        
    > The purpose of this function is to audio/peaks in the spectrogram. While iterating over the spectrogram when a peak is detected, it should be marked on an identically sized matrix with 1. It's best to create an identically sized matrix filled with 0s and then add the 1s.
        
    Input:    
    - a 2d array  which represents a transposed spectrogram. Input format is (frequency_bins, time_steps).
    - additional parameters which are required inside of the function (set by user)

    
    Output: 
    - a 2d array which represents detected peaks with 1s and non events with 0s. Output format is (frequency_bins, time_steps).
    
    ----

    #### 3. peak clustering function (`fn_cluster_audio_events(spectrogram_peaks, event_detection_parameters)`):
        
    > The purpose of this function is to cluster the peaks into "complete" events
        
    Input:    
    - a 2d array which represents detected peaks with 1s and non events with 0s. Output format is (frequency_bins, time_steps).
    - additional parameters which are required inside of the function (set by user)
    
    Output: 
    - a 2d array  which represents a spectrogram. Output format is (frequency_bins, time_steps). 


## Jupyter Notebooks

notebooks/

* ### audio_sample_analysis.ipynb

    Use this notebook to quickly checkout the wave plot and mel spectrogram of downloaded samples.


## Project Structure:
```


├──data/
│   ├──xeno_canto_bsfatw/  
│   │   ├──occurrence.txt  
│   │   ├──...
│   │   └──multimedia.txt
│   │
│   ├──data_dictionary/
│   │   ├──species_info.csv
│   │   ├──species_sample_info.json
│   │   └──samples_metadata.json
│   │
│   ├──raw/
│   ├──processed/
│   └──...
│ 
├──songbird/
│   ├──data/ 
│   │   ├── create_data_dictionary.py  
│   │   └── download_data.py
│   ├──audio/ 
│   │   ├── create_data_dictionary.py  
│   │   └── download_data.py
│ 
└── notebooks/
```

Dictionary in samples_metadata.json
```python
sample_dict = {
    'gbifID' : int, #unique gbifID 
    'recording_link' : str, #link to media
    'decimal_latitude' : float,
    'decimal_longitude': float,
    'date' : datetime_string,
    'behavior' : str[], # > array of strings 
    'forefront' : int,
    'background_birds' : int[], # > array of species key ints 
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