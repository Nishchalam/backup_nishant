import numpy as np
import os
import shutil



path = "/speech/nishanth/root_exps/data"
dst_path = '/speech/nishanth/raw_exps/5_fold_rev_final_data_8khz'
base_path = '/speech/nishanth/raw_exps/full_frames_crepe_16khz'
noises = ['0', '5', '10', '20', 'clean']
splits = [ 'test']
gender_dict = {"M":"MALE", "F":"FEMALE"}   
def hz_to_cent(f):
    mask = np.where(f==0.0)
    
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent -= (1997.37 - 20)
    cent[mask] = 0.0
    return np.round(cent)

def hz_to_bin(f):
    mask = np.where(f == 0.0)
    
    cent = 1200 * np.log2((f / 10) + 1e-9)
    cent -= (1997.37 - 20)
    cent[mask] = 0.0
    bin_ = np.floor(cent / 20)
    
    return np.minimum(bin_, 299)  # Ensure bin index is within [0, 299]

def one_hot_encode(bin_indices, num_bins=300):
    # Convert the bin indices to integers
    bin_indices = bin_indices.astype(int)
    
    # Initialize an array of zeros with shape (len(bin_indices), num_bins)
    one_hot = np.zeros((len(bin_indices), num_bins))
    
    # Set the appropriate bins to 1
    one_hot[np.arange(len(bin_indices)), bin_indices] = 1
    
    return one_hot




for s in splits:
    if s != "test":
        labels = os.listdir(f"{dst_path}/{s}/labels")

        label_path_new = f"{dst_path}/{s}/labels_one_hot"
        os.makedirs(label_path_new, exist_ok = True)
        for file in labels:
            res = np.load(os.path.join(f"{dst_path}/{s}/labels", file))

            bin_indices = hz_to_bin(res)
            one_hot_labels = one_hot_encode(bin_indices)
            #res = res[0,:,:]
            np.save(os.path.join(f"{dst_path}/{s}/labels_one_hot", file), one_hot_labels)
    
    else:
        for n in noises:
            labels = os.listdir(f"{dst_path}/{s}/{n}/labels")

            label_path_new = f"{dst_path}/{s}/{n}/labels_cent"
            os.makedirs(label_path_new, exist_ok = True)
            for file in labels:
                res = np.load(os.path.join(f"{dst_path}/{s}/{n}/labels", file))
                cent = hz_to_cent(res)
                # bin_indices = hz_to_bin(res)
                # one_hot_labels = one_hot_encode(bin_indices)
                np.save(os.path.join(f"{dst_path}/{s}/{n}/labels_cent", file), cent)
           