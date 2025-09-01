import os
import numpy as np
from tqdm import tqdm

data_path = "/speech/nishant/raw_feature_exps/frames"

for db in [0, 5, 10, 20, "clean"]:
    test_path = os.path.join(data_path, str(db), "test")
    label_path = os.path.join(data_path, str(db), "test_label")
    new_test_path = os.path.join(data_path, str(db), "test_new")

    os.makedirs(new_test_path, exist_ok=True)

    label_files = os.listdir(label_path)
    for file_name in tqdm(label_files):
        arr = np.load(os.path.join(test_path, file_name))
        labels = np.load(os.path.join(label_path, file_name))


        len_diff = arr.shape[0] - labels.shape[0]
        if arr.shape[0] > labels.shape[0]:
            new_gd_arr = arr[:labels.shape[0], :]
            np.save(os.path.join(new_test_path, file_name), new_gd_arr)
        else:
            new_gd_arr = np.concatenate((arr, np.zeros((labels.shape[0] - arr.shape[0], arr.shape[1]))), axis=0)
            np.save(os.path.join(new_test_path, file_name), new_gd_arr)

    # old_test = np.load("/speech/nishant/raw_feature_exps/final_data/test/mic_F01_sa1.npy")
    # print(old_test.shape)
    # new_test = np.load("/speech/nishant/raw_feature_exps/final_data/test_new/mic_F01_sa1.npy")
    # print(new_test.shape)
    # label = np.load("/speech/nishant/raw_feature_exps/final_data/test_label/mic_F01_sa1.npy")
    # print(label.shape)
