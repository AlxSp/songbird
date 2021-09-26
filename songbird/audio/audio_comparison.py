#%%
import pydub
import time
import numpy as np
import pandas as pd
import json
import os
import audio_processing as ap
from scipy import signal
from skimage.feature import peak_local_max
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import hashlib

#%%
sample_ids = [
    2243804495,
    2432423171,
    2243742980,
    2243784966,
    2243675399,
    2455252743,
    2432420110,
    2432417551,
    2243570965,
    2243667228,
    2243883806,
    2243740447,
    2243583779,
    2432421155,
    2243571495,
    2243587373,
    2243778866,
]

#%%
def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks 

#%%
def get_constellations(peak_coordinates, step_size, window_dim):
    constellation_x_coordinates_arr = []
    constellation_y_coordinates_arr = []
    for anchor_coordinate in peak_coordinates:
        anchor_x_index = anchor_coordinate[0]
        anchor_y_index = anchor_coordinate[1]

        window_start = anchor_x_index + step_size 
        window_end = window_start + window_dim[0]
        window_top = anchor_y_index + window_dim[1] // 2
        window_bottom = anchor_y_index - window_dim[1] // 2
        
        are_in_window_x = (peak_coordinates[:,0] > window_start) & (peak_coordinates[:,0] < window_end)
        are_in_window_y = (peak_coordinates[:,1] > window_bottom) & (peak_coordinates[:,1] < window_top)

        in_constellation_indices = are_in_window_x & are_in_window_y

        constellation_coordinates = peak_coordinates[in_constellation_indices]
        constellation_pair_num = len(constellation_coordinates)

        x_coordinates = np.zeros(2 * constellation_pair_num)
        y_coordinates = np.zeros(2 * constellation_pair_num)

        x_coordinates[slice(1, None, 2)] = constellation_coordinates[:,0]
        y_coordinates[slice(1, None, 2)] = constellation_coordinates[:,1]

        x_coordinates[slice(0, None, 2)] = anchor_coordinate[0]
        y_coordinates[slice(0, None, 2)] = anchor_coordinate[1]

        constellation_x_coordinates_arr.append(x_coordinates)
        constellation_y_coordinates_arr.append(y_coordinates)
        
    x_coordinates = np.concatenate(constellation_x_coordinates_arr)
    y_coordinates = np.concatenate(constellation_y_coordinates_arr)

    return x_coordinates, y_coordinates
        #constellation_coordinates_arr.append()
        #np.full((len(constellation_coordinates, 4), anchor_coordinate))


    

def get_constellation_hashes(peak_coordinates, step_size, window_dim, sec_per_step, rfft_freq_bins):

    #def compute_hash(arr):
    constellation_hashes = []
    for anchor_coordinate in peak_coordinates:
        anchor_x_index = anchor_coordinate[0]
        anchor_y_index = anchor_coordinate[1]

        window_start = anchor_x_index + step_size 
        window_end = window_start + window_dim[0]
        window_top = anchor_y_index + window_dim[1] // 2
        window_bottom = anchor_y_index - window_dim[1] // 2
        
        are_in_window_x = (peak_coordinates[:,0] > window_start) & (peak_coordinates[:,0] < window_end)
        are_in_window_y = (peak_coordinates[:,1] > window_bottom) & (peak_coordinates[:,1] < window_top)

        in_constellation_indices = are_in_window_x & are_in_window_y

        constellation_coordinates = peak_coordinates[in_constellation_indices]
        constellation_time_differences = (constellation_coordinates[:,0] - anchor_x_index) * sec_per_step
        constellation_frequencies = rfft_freq_bins[constellation_coordinates[:,1]]

        anchor_frequency = rfft_freq_bins[anchor_y_index]
        anchor_time = anchor_x_index * sec_per_step

        for i in range(len(constellation_coordinates)):
            m = hashlib.md5()
            m.update(anchor_frequency)
            m.update(constellation_frequencies[i])
            m.update(constellation_time_differences[i])
            constellation_hashes.append((anchor_time, m.digest()))

    return constellation_hashes
    #    print(f"constellation parameters: anchor fre:{anchor_frequency}, consti freq: {constellation_frequencies}, consti time diff: {constellation_time_differences}")
    print("Coordinates found")
    print(len(constellation_hashes))
        
        #print(peak_coordinates[in_constellation_indices])

#%% 
# def hash(peak_frequencies, Peak_time_difference):


#%%
def intersect(a_arr, b_arr):
    cumdims = (np.maximum(a_arr.max(),b_arr.max())+1)**np.arange(b_arr.shape[1])
    return np.in1d(a_arr.dot(cumdims),b_arr.dot(cumdims))
#%%
def create_finger_prints(sample_id, audio_conversion_parameters, plot = False):
    
    if plot:
        ap.empty_or_create_dir(os.path.join(ap.plots_path, str(sample_id)))

    sample, _ = ap.load_audio_sample(sample_id, audio_conversion_parameters.sample_rate)

    sos = signal.butter(10, 3500, 'hp', fs=audio_conversion_parameters.sample_rate, output='sos')

    sample = signal.sosfilt(sos, sample)

    spectrogram, rfft_bin_freq = ap.create_spectrogram(sample, audio_conversion_parameters.window_size, audio_conversion_parameters.step_size, audio_conversion_parameters.sample_rate)

    spectrogram, rfft_bin_freq = ap.trim_to_frequency_range(spectrogram, rfft_bin_freq, audio_conversion_parameters.max_frequency, audio_conversion_parameters.min_frequency)

    spectrogram = spectrogram.numpy()
    #peaks = detect_peaks(spectrogram)
    if plot:
        ap.save_spectrogram_plot(spectrogram.T, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram (Decibel)", y_labels = rfft_bin_freq)
    
    maximums = maximum_filter(spectrogram, size=20)

    if plot:
        ap.save_spectrogram_plot(maximums.T, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram (Decibel)", y_labels = rfft_bin_freq)


    # print(maximums.shape)     
    # print(spectrogram.shape)

    std_value_1 = 1.2
    #std_value_2 = 1.3
    mask = maximums > np.mean(maximums) + np.std(maximums) * std_value_1 #> -30#np.mean(filtered, axis=0)
    #mask2 = maximums > np.mean(maximums, axis = 0) + np.std(maximums, axis = 0) * std_value_2

    peak_coordinates = peak_local_max(spectrogram, min_distance=7)
    mask_coordinates = np.transpose(np.nonzero(mask))

    peak_coordinates = peak_coordinates[intersect(peak_coordinates, mask_coordinates)]

    sec_per_step = audio_conversion_parameters.step_size / audio_conversion_parameters.sample_rate
    sample_constellation_hashes = get_constellation_hashes(peak_coordinates, 10, (50,50), sec_per_step, rfft_bin_freq)
    
    if plot:
        x_coordinates, y_coordinates = get_constellations(peak_coordinates, 10, (50,50))
        ap.save_spectrogram_plot(mask.T, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Mask 1", y_labels = rfft_bin_freq)
        #ap.save_spectrogram_plot(mask2.T, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Mask 2", y_labels = rfft_bin_freq)
        #ap.save_spectrogram_plot(clean.T, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram with Mask", y_labels = rfft_bin_freq)
        ap.save_spectrogram_plot_with_peaks(spectrogram.T, peak_coordinates, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram with Mask 2", y_labels = rfft_bin_freq)
        ap.save_spectrogram_plot_with_constellations(spectrogram.T, peak_coordinates, x_coordinates, y_coordinates, audio_conversion_parameters.sample_rate, audio_conversion_parameters.step_size, sample_id, title = "Spectogram with Constellations", y_labels = rfft_bin_freq)
    
    return sample_constellation_hashes
    # x_coordinates, y_coordinates = get_constellations(peak_coordinates, 10, (50,50))
    # print(x_coordinates)

    # #peak_coordinates = np.asarray([coordinate for coordinate in peak_coordinates if coordinate in mask_coordinates])
    # # print(peak_coordinates)
    # #coordinates = peak_local_max(mask, min_distance=20)

    # clean = spectrogram * mask
    #clean2 = spectrogram * mask2

#%%
for sample_id in sample_ids:
    pass

#%%
class AudioConversionParameter:
    def __init__(self) -> None:
        self.window_size = 2048
        self.step_size = 512
        self.sample_rate  = 44100
        self.max_frequency = 10000
        self.min_frequency = 2500

acp = AudioConversionParameter()

#%%
start_time = time.time()
sample_id = 2243759963 #2243804495 #2243570965 #2243778866
print(f"sample id : {sample_id}")
create_finger_prints(sample_id, acp, True)  
end_time = time.time()
print(f"runtime: {end_time-start_time}s")

#%%
with open(os.path.join("/home/alex/Dev/songbird/data/data_dictionary/download_species_sample_info.json"), "r") as json_reader:
    data = json.load(json_reader)
species_samples = data["2473663"]["forefront_sample_ids"]

#%%
species_samples
#%%
all_samples_df = pd.DataFrame(columns = ["sample_id", "time_in_sample", "hash"])

sample_arr = species_samples#[2243804495, 2243570965, 2243778866]
start_time = time.time()
for i, sample_id in enumerate(sample_arr):
    print(f"{i+1}/{len(sample_arr)}")
    sample_hashes = create_finger_prints(sample_id, acp)
    hash_dict = {
        "sample_id" : [], 
        "time_in_sample" : [], 
        "hash" : []
    }

    for hash_data in sample_hashes:
        hash_dict["sample_id"].append(sample_id)
        hash_dict["time_in_sample"].append(hash_data[0])
        hash_dict["hash"].append(hash_data[1])
    sample_df = pd.DataFrame(data = hash_dict)
    all_samples_df = all_samples_df.append(sample_df)
    
end_time = time.time()
print(f"Sample num {len(sample_arr)} runtime: {end_time-start_time}s")

# %%
all_samples_df

#%%
duplicates = all_samples_df.pivot_table(index=['hash'], aggfunc='size')
hash_has_duplicate = duplicates.values > 1
matching_hashes = duplicates[hash_has_duplicate].index.tolist()
#%%
samples_with_matches_df = all_samples_df[all_samples_df["hash"].isin(matching_hashes)]

#len(all_samples_df["hash"][0].values[0].decode("utf-8"))

#%%
print(f'Processed samples count: {len(all_samples_df["sample_id"].unique())}')
print(f'Processed samples with matching hashes count: {len(samples_with_matches_df["sample_id"].unique())}')
print(f'Samples with no matches: {[sample_id for sample_id in all_samples_df["sample_id"].unique() if sample_id not in samples_with_matches_df["sample_id"].unique()]}')

#%%
def get_sample_hash_pairs(samples_df, sample_id):
    sample_df = samples_df[samples_df["sample_id"] == sample_id] 

    sample_df = sample_df.sort_values(by="time_in_sample")

    sample_hash_list =  sample_df["hash"].tolist()
    sample_time_list = sample_df["time_in_sample"].tolist()

    hash_pairs = [] #(hash1, hash2, time_difference)
    for index in range(len(sample_time_list[:-1])):
        current_hash = sample_hash_list[index]
        current_hash_time = sample_time_list[index]
        hash_to_pair_index = index + 1
        
        next_hash_time = sample_time_list[hash_to_pair_index]
        while next_hash_time == current_hash_time:
            hash_to_pair_index += 1
            next_hash_time = sample_time_list[hash_to_pair_index]

        for jndex in range(hash_to_pair_index, len(sample_time_list)):
            hash_time = sample_time_list[jndex] 
            if hash_time == next_hash_time:
                time_difference = hash_time - current_hash_time
                hash_pairs.append([current_hash, sample_hash_list[jndex], time_difference])

    return hash_pairs

#%%
sample_id = 2243804495
sample_pairs = get_sample_hash_pairs(samples_with_matches_df, sample_id)
#%%
sample_pairs
#hash_pairs
# hash_list
# sample_df
# samples_with_matches_df[samples_with_matches_df["hash"].isin(sample_df["hash"].tolist())]["sample_id"].unique()

#%%
sample_unique_hashes = sample_df["hash"].unique().tolist()
other_samples_df = samples_with_matches_df[(samples_with_matches_df["hash"].isin(sample_unique_hashes)) & (samples_with_matches_df["sample_id"] != sample_id)] 
#%%
for sample_id, sample_data_df in other_samples_df.groupby(by="sample_id"):
    sample_data_df = sample_data_df[sample_data_df["hash"].isin(sample_unique_hashes)]
    sample_data_df = sample_data_df.sort_values(by="time_in_sample")
    print(sample_id)
    print(sample_data_df)
    matched_pairs = []
    # hash_list = sample_data_df["hash"].tolist()
    # time_list = sample_data_df["time_in_sample"].tolist()

    for hash_pair_data in sample_pairs:
        hash_1 = sample_data_df[sample_data_df["hash"] == hash_pair_data[0]]["time_in_sample"].to_numpy()
        hash_2 = sample_data_df[sample_data_df["hash"] == hash_pair_data[1]]["time_in_sample"].to_numpy()
        if not hash_1.any() or not hash_2.any():
            continue
        else:
            pair_hash_differences = hash_2 - hash_1[:, np.newaxis]
            print(pair_hash_differences) 
    # for index, row in sample_data_df.iter_rows(): 
    # print(sample_data_df)






# #%%
# A = np.array([[1, 1, 1],
#        [1, 1, 2],
#        [1, 1, 3],
#        [1, 1, 4]])

# B = np.array([[0, 0, 0],
#        [1, 0, 2],
#        [1, 0, 3],
#        [1, 0, 4],
#        [1, 1, 0],
#        [1, 1, 1],
#        [1, 1, 4]])

# #%%
# cumdims = (np.maximum(A.max(),B.max())+1)**np.arange(B.shape[1])
# #%%
# cumdims = np.arange(1, B.shape[1]+1)
# #%%
# cumdims
# # %%
# A.dot(cumdims)
# # %%
# B.dot(cumdims)
# # %%
# np.in1d(A.dot(cumdims),B.dot(cumdims))
# # %%
# A[np.in1d(A.dot(cumdims),B.dot(cumdims))]


# %%
