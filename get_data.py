
import xenocanto
import pandas as pd
import numpy as np
import os


species_path = os.path.join("dataset",'species_info.csv') 
species_info_df = pd.read_csv(species_path)

rec_min = 250

print("Total number of species:", len(species_info_df))

min_species_info_df = species_info_df[species_info_df['forefront recs'] > rec_min]

print("Number of species meeting min:", len(min_species_info_df))

print(min_species_info_df.head())

#min_species_info_df

xenocanto.metadata()
#xenocanto.download()



