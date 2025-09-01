import numpy as np
import os
import shutil




dst_path = '/speech/nishant/clean_research/final_data'
noises = ['0', '5', '10', '20', 'clean']
splits = ['train', 'valid', 'test']

for s in splits:
    if s != "test":
        label_files = os.listdir(os.path.join(dst_path, s ,"labels"))
        for file_name in label_files:
            #new_file_name = f"1{file_name}"
            os.makedirs(os.path.join(dst_path, s, "label_flip"),exist_ok= True)
            os.makedirs(os.path.join(dst_path, s, "gd_1.008_flip"),exist_ok= True)
            gd_arr = np.load(os.path.join(dst_path, s , "gd_1.008", file_name))
            labels = np.load(os.path.join(dst_path, s,  "labels", file_name))
            #new_gd_arr = np.flip(gd_arr, axis =1)
            np.save(os.path.join(dst_path, s, "label_flip" ,file_name), labels)
            np.save(os.path.join(dst_path, s, "gd_1.008_flip" ,file_name), gd_arr)
            

            
    else:
        for q in noises:
            label_files = os.listdir(os.path.join(dst_path, s ,q ,"labels"))
            for file_name in label_files:
                #new_file_name = f"1{file_name}"
                os.makedirs(os.path.join(dst_path, s, q, "label_flip"), exist_ok= True)
                os.makedirs(os.path.join(dst_path, s, q, "gd_1.008_flip"),exist_ok= True)
                gd_arr = np.load(os.path.join(dst_path, s , q,  "gd_1.008", file_name))
                labels = np.load(os.path.join(dst_path, s, q ,"labels", file_name))
                #new_gd_arr = np.flip(gd_arr, axis =1)
                np.save(os.path.join(dst_path, s,q, "label_flip" ,file_name), labels)
                np.save(os.path.join(dst_path, s,q, "gd_1.008_flip" ,file_name), gd_arr)
                