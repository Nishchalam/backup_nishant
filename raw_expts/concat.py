import numpy as np
import os

def load_and_concatenate_npy_files(directory_path):
    # List all .npy files in the directory
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]
    npy_files.sort()  # Ensure files are in the desired order

    # List to hold all loaded arrays
    all_arrays = []
    
    for file in npy_files:
        file_path = os.path.join(directory_path, file)
        array = np.load(file_path)
        all_arrays.append(array)

    # Concatenate arrays along a new dimension (axis=0)
    concatenated_array = np.concatenate(all_arrays, axis=0)
    
    return concatenated_array
save_path = "/speech/nishanth/raw_exps/5_fold_rev_final_data_8khz/test/5/concat_cents"
os.makedirs(save_path,exist_ok=True)
def save_array(array, save_path):
    np.save(save_path, array)
    print(f"Array saved to {save_path}")
# Usage
directory_path = '/speech/nishanth/raw_exps/5_fold_rev_final_data_8khz/test/5/labels_cent'
concatenated_array = load_and_concatenate_npy_files(directory_path)
save_array(concatenated_array, save_path)
print(concatenated_array.shape)  # Check the shape of the concatenated array
# import numpy as np
# input_file = '/speech/nishanth/raw_exps/5_fold_rev_final_data_8khz/train/concat_frames.npy'  # Replace with your input file path
#   # Replace with your desired output file path

# # Load the 2D array
# array_2d = np.load(input_file)

# # Reshape the 2D array to 3D
# array_3d = array_2d.reshape((5140628, 256, 1))
# np.save(input_file, array_3d)

# print(array_3d.shape)  # Should print (12845165, 1024, 1)
