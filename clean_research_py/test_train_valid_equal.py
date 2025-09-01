import os
import shutil
import random

# Define paths
src_path = '/speech/nishanth/clean_research/new_group_delay'
noises = ['0', '5', '10', '20', 'clean']
splits = ['train_eq', 'test_eq', 'valid_eq']

# Define split ratios
train_ratio = 0.6
test_ratio = 0.2
valid_ratio = 0.2

for noise in noises:
    # Path to the 'all' directory
    all_path = os.path.join(src_path, noise, 'all')
    files = os.listdir(all_path)

    # Calculate the number of files for each split
    total_files = len(files)
    train_split = int(train_ratio * total_files)
    test_split = int(test_ratio * total_files)
    valid_split = total_files - train_split - test_split  # Ensures all files are used

    # Create destination directories if they don't exist
    for split in splits:
        os.makedirs(os.path.join(src_path, noise, split), exist_ok=True)

    # Split the files
    train_files = files[:train_split]
    test_files = files[train_split:train_split + test_split]
    valid_files = files[train_split + test_split:]

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(all_path, file), os.path.join(src_path, noise, 'train_eq', file))

    for file in test_files:
        shutil.copy(os.path.join(all_path, file), os.path.join(src_path, noise, 'test_eq', file))

    for file in valid_files:
        shutil.copy(os.path.join(all_path, file), os.path.join(src_path, noise, 'valid_eq', file))

print("Files have been successfully split and copied into train, test, and valid directories.")
