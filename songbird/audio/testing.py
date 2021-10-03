import numpy as np

sample_length = 4

event_start = 0
event_end = 7

sliding_win_indices = []
for i in range(event_start, event_end - sample_length + 1, 1):
    sliding_win_indices.append([i, i + sample_length])


start_indices = np.arange(event_start, event_end - sample_length  + 1)
end_indices = np.arange(event_start + sample_length, event_end  + 1)

sliding_win_indices = np.c_[start_indices, end_indices]#np.meshgrid(start_indices, end_indices)

sliding_win_indices = np.concatenate((sliding_win_indices, np.array([[1,5]])), axis =0)

print(len(sliding_win_indices))

sliding_win_indices = np.unique(sliding_win_indices, axis = 0)

print(len(sliding_win_indices))

# sliding_win_indices = np.concatenate(
#     ( start_indices,
#       end_indices)
#     # , axis = 1
#     )#np.mgrid[event_start:event_end - sample_length + 1, event_start + sample_length:100 + 1] 

for win in sliding_win_indices:
    print(win[0], win[1])