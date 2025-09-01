# import numpy as np
# import os
# import shutil




# dst_path = '/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008'
# base_path = '/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008'
# noises = ['0', '5', '10', '20', 'clean']
# splits = [ 'train','test',"valid"]

# ######## FIX LENGTH #######
# for s in splits:
#     if s != "test":
#         os.makedirs(os.path.join(dst_path, s ,"gd_norm"), exist_ok = True)
#         gd = os.listdir(os.path.join(dst_path, s ,"gd_1.008"))
#         for file_name in gd:
#             data = np.load(os.path.join(dst_path, s , "gd_1.008", file_name))
#             mean = np.mean(data)
#             new_data = data - mean
#             print(new_data.shape)
#             auto_corr = np.correlate(new_data,new_data, MODE = 'FULL')
#             auto_corr = auto_corr[len(auto_corr)/2:]
# #             data_min = np.min(data, axis=2, keepdims=True)
# #             data_max = np.max(data, axis=2, keepdims=True)

# # # Avoid division by zero by adding a small value
# #             normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
#             np.save(os.path.join(dst_path, s ,"gd_norm", file_name), auto_corr)
            

#     else:
#         for q in noises:
#             os.makedirs(os.path.join(dst_path, s ,q ,"gd_norm"),exist_ok = True)
#             gd = os.listdir(os.path.join(dst_path, s ,q,"gd_1.008"))
#             for file_name in gd:
#                 data = np.load(os.path.join(dst_path, s ,q, "gd_1.008", file_name))
#                 mean = np.mean(data)
#                 new_data = data - mean
#                 new_data = data - mean
#                 auto_corr = np.correlate(new_data,new_data, MODE = 'FULL')
#                 auto_corr = auto_corr[len(auto_corr)/2:]
#                 # data_min = np.min(data, axis=2, keepdims=True)
#                 # data_max = np.max(data, axis=2, keepdims=True)

#     # Avoid division by zero by adding a small value
#                 # normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
#                 np.save(os.path.join(dst_path, s, q ,"gd_norm", file_name), auto_corr)
                                
import numpy as np
import os
import shutil




dst_path = '/speech/nishanth/mir_dataset/final_data'
base_path = '/speech/nishanth/mir_dataset/final_data'
noises = ['0', '5', '10', '20', 'clean']
splits = [ 'train','test',"valid"]

######## FIX LENGTH #######
for s in splits:
    if s != "test":
        os.makedirs(os.path.join(dst_path, s ,"gd_thres"), exist_ok = True)
        gd = os.listdir(os.path.join(dst_path, s ,"gd_all_radii"))
        for file_name in gd:
            data = np.load(os.path.join(dst_path, s , "gd_all_radii", file_name))
            data = data[2,:,:512]
            mean = np.mean(data)
            new_data = data - mean
            # print(new_data.shape)
            # new_data[new_data<0] = 0
            np.save(os.path.join(dst_path, s ,"gd_thres", file_name), new_data)
            

    else:
        for q in noises:
            os.makedirs(os.path.join(dst_path, s ,q ,"gd_thres"),exist_ok = True)
            gd = os.listdir(os.path.join(dst_path, s ,q,"gd_all_radii"))
            for file_name in gd:
                data = np.load(os.path.join(dst_path, s ,q, "gd_all_radii", file_name))
                data = data[2,:,:512]
                mean = np.mean(data)
                new_data = data - mean
                # new_data[new_data<0] = 0

    # Avoid division by zero by adding a small value
                # normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
                np.save(os.path.join(dst_path, s, q ,"gd_thres", file_name), new_data)
                                