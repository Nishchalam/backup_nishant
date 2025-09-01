import os
import numpy as np
import shutil

src_path = '/speech/nishanth/clean_research/new_group_delay'


noises = ['0','5','10','20','clean']
splits = ['train','test','valid']


for i in noises:
    for j in splits:
        src1_path = os.path.join(src_path,i,j)
        dst_path1 = os.path.join(src_path,i, "all")
        os.makedirs(dst_path1, exist_ok = True)
        files = os.listdir(src1_path)
        for file in files:
            src = os.path.join(src1_path, file)
            dst = os.path.join(dst_path1, file)
            shutil.copy(src, dst)        