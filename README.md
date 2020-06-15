# Songbird Project

## Setup: 
1. Download the dataset: [Xeno-canto - Bird sounds from around the world](https://www.gbif.org/dataset/b1047888-ae52-4179-9dd5-5448ea342a24#methodology)
1. Extract it into the xeno_canto_bsfatw directory
1. run create_data_dictionary.py
    - to check out info on the dataset run the meta_data_analysis.ipynb notebook
1. run download_data.py (Currently downloads only 1 sample)
1. run audio_analysis for info on the sample 

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
│   │   └──samples_info
│   │       ├── species_key_0.json
│   │       ├── ...
│   │       └── species_key_N.json
│   ├──raw
│
├── X    
└── X2
```


```python
sample dict:

sample_dict = {
    'gbifID' : int, #unique gbifID 
    'recording_link' : string, #link to media
    'decimal_latitude' : float,
    'decimal_longitude': float,
    'date' : datetime_string,
    'behavior' : [], # > array of strings 
    'background_birds' : [], # > array of species key ints 
}
```