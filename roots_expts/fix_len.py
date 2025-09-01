import os
import numpy as np

path = "/speech/nishant/clean_research/residual_exps/all_final_data/test"


label_files = os.listdir(os.path.join(path, "labels"))
for file_name in label_files:
    gd_arr = np.load(os.path.join(path, "residual", file_name))
    labels = np.load(os.path.join(path, "labels", file_name))

    print(gd_arr.shape, labels.shape)

    len_diff = gd_arr.shape[1] - labels.shape[0]
    if gd_arr.shape[0] > labels.shape[0]:
        new_gd_arr = gd_arr[:labels.shape[0], :]
        np.save(os.path.join(path, "residual", file_name), new_gd_arr)
    else:
        new_gd_arr = np.concatenate((gd_arr, np.zeros((labels.shape[0] - gd_arr.shape[0], 255))), axis=1)
        np.save(os.path.join(path, "residual", file_name), new_gd_arr)

