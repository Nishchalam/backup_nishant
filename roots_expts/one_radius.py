import numpy as np
import os
os.makedirs("/speech/nishant/roots_exps/final_data/train/new_group_delay")
path = "/speech/nishant/roots_exps/final_data/train/group_delay"
arr = os.listdir(path)
# print(arr)
gd_files =[os.path.join(path , x) for x in arr]
# data_list =[]
for matrix in arr:
    data = np.load(os.path.join(path,matrix))
    new_data = data[0,:,:]
    # print(data.shape)
    # print(new_data.shape)
    # np.shape(data)
    # data_list.append(new_data)
    new_path = os.path.join("/speech/nishant/roots_exps/final_data/train/new_group_delay", matrix)
    np.save(new_path, new_data)
