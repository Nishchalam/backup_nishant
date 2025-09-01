import numpy as np
import os
os.makedirs("/speech/nishant/clean_research/final_data/test/gd_1.002")
path = "/speech/nishant/clean_research/final_data/test/group_delay"
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
    new_path = os.path.join("/speech/nishant/clean_research/final_data/test/gd_1.002", matrix)
    np.save(new_path, new_data)
