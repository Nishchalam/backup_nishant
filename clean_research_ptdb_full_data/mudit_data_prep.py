import numpy as np
import os
import shutil



import os
import shutil
import random

snr_levels = ['0', '5', '10', '20', "clean"]  # Example SNR levels
base_dir = '/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2'
output_dir = '/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2_mudit'

for snr in snr_levels:
    snr_path = os.path.join(base_dir, snr)
    
    for split in ['test', 'train', 'valid']:
        split_path = os.path.join(snr_path, split)
        new_split_path = os.path.join(output_dir, snr, split)
        os.makedirs(new_split_path, exist_ok=True)
        
        for filename in os.listdir(split_path):
            # Determine the new folder based on the filename
            if any(x in filename for x in ['M09', 'M10', 'F09', 'F10']):
                new_folder = 'test'
            else:
                new_folder = 'train'
                
            # Create the new directory if it doesn't exist
            new_dir = os.path.join(output_dir, snr, new_folder)
            os.makedirs(new_dir, exist_ok=True)
            
            # Copy the file to the new directory
            src_file = os.path.join(split_path, filename)
            dest_file = os.path.join(new_dir, filename)
            shutil.copy(src_file, dest_file)

            print(f"Copied {filename} to {new_folder} folder")
    
    # Further split the remaining train set into 80% train and 20% valid
    train_dir = os.path.join(output_dir, snr, 'train')
    train_files = os.listdir(train_dir)
    random.shuffle(train_files)
    
    # Determine the split index
    split_idx = int(len(train_files) * 0.8)
    
    new_valid_dir = os.path.join(output_dir, snr, 'valid')
    os.makedirs(new_valid_dir, exist_ok=True)
    
    # Move 20% of the files to the valid set
    for filename in train_files[split_idx:]:
        src_file = os.path.join(train_dir, filename)
        dest_file = os.path.join(new_valid_dir, filename)
        shutil.move(src_file, dest_file)
        print(f"Moved {filename} to valid folder")
        
print("File copying and splitting completed.")
