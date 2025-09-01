import numpy as np

# File paths for input data
file_path = '/speech/nishanth/clean_research/ptdb_full_data/full_final_data/train/labels/5_mic_M01_sa1.npy' 
file_path_2 = "/speech/nishanth/clean_research/ptdb_full_data/voiced_unvoiced_array.npy"

# Load the data from both files
data1 = np.load(file_path)
data2 = np.load(file_path_2)

# Print the shape of both arrays
print("Shape of data1:", np.shape(data1))
print("Shape of data2:", np.shape(data2))

# Find the minimum length of the two arrays
min_length = min(len(data1), len(data2))

# Slice both arrays to the minimum length
data1_sliced = data1[:min_length]
data2_sliced = data2[:min_length]

# Combine the two sliced arrays column-wise
new_array = np.column_stack((data1_sliced, data2_sliced))

# Save the combined array to a text file
text_file_path = '/speech/nishanth/clean_research/file.txt'  # Output file path
np.savetxt(text_file_path, new_array, fmt='%d')

print(f"Combined array saved to {text_file_path}")



# def median_filter(correct, pred, filter_length):

#     assert len(pred) == len(correct)
#     filtered_data  = pred
#     shift = 1
#     # no_of_frame = int(len(pred)/shift - filter_length)
#     for i in range(len(pred)):
#         if correct[i] != 0 and correct[i+filter_length] != 0:
#             window = pred[i : filter_length+i]
#             median_value = torch.median(window)
#             filtered_data[i] = median_value
#         else:
#             continue
#     return filtered_data