import numpy as np
import os
import shutil

src_path = "/speech/nishant/raw_feature_exps/final_data"
dst_path = "/speech/nishant/raw_feature_exps/5_fold_rev_final_data_8khz"
noises = ['0', '5', '10', '20', 'clean']
splits = ['train', 'valid', 'test']
for s in splits:
    if s != "test":
        label_files = os.listdir(os.path.join(src_path, s ,"labels"))
        for file_name in label_files:
            src1 = os.path.join(src_path, s , "raw", file_name)
            src2 = os.path.join(src_path, s,  "labels", file_name)
            dst1 = os.path.join(dst_path, s , "raw", file_name)
            dst2 = os.path.join(dst_path, s , "labels", file_name)
            shutil.copy(src1,dst1)
            shutil.copy(src2,dst2)
    else:
        for q in noises:
            label_files = os.listdir(os.path.join(src_path, s ,q ,"labels"))
            for file_name in label_files:
                src3 = os.path.join(src_path, s ,q, "raw", file_name)
                src4 = os.path.join(src_path, s, q,  "labels", file_name)
                dst3 = os.path.join(dst_path, s ,q, "raw", file_name)
                dst4 = os.path.join(dst_path, s ,q, "labels", file_name)
                shutil.copy(src3,dst3)
                shutil.copy(src4,dst4)               
                                               