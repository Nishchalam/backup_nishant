import numpy as np
import os

base_path = '/speech/nishant/clean_research/final_data'
splits = ['train', 'valid', 'test']
for i in splits:
    path = os.path.join(base_path, i)
    data_path = os.path.join(path, 'gd_1.002')
    new_data_path = os.path.join(path, 'new_gd_1.002')
    os.makedirs(new_data_path, exist_ok=True)
    files = os.listdir(data_path)
    for file in files:
        file_path = os.path.join(data_path, file)
        array = np.load(file_path)
        mean_value = np.mean(array)
        array = array - mean_value
        norm_value = np.linalg.norm(array)
        normalized_array = array/norm_value
        normalized_array[normalized_array < -1] = 0
        np.save(os.path.join(new_data_path, file), array)