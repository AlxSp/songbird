# Songbird Project

## Setup: 
1. Download the dataset: [Xeno-canto - Bird sounds from around the world](https://www.gbif.org/dataset/b1047888-ae52-4179-9dd5-5448ea342a24#methodology)
1. Extract it into the xeno_canto_bsfatw dir

## Project Structure:
```
├──dataset
│   ├── x  
│   └── species_info.csv  
├──xeno_canto_bsfatw  
│   ├── occurrence.txt  
│   ├── ...
│   └── multimedia.txt  
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