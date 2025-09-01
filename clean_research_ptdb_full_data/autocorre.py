# import numpy as np
# import os
# import shutil
# from scipy import signal
# from tqdm import tqdm



# dst_path = '/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008'
# base_path = '/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008'
# noises = ['0', '5', '10', '20', 'clean']
# splits = [ 'train','test',"valid"]

# ######## FIX LENGTH #######
# for s in splits:
#     if s != "test":
#         os.makedirs(os.path.join(dst_path, s ,"gd_thres"), exist_ok = True)
#         gd = os.listdir(os.path.join(dst_path, s ,"gd_1.008"))
#         for file_name in tqdm(gd):
#             data = np.load(os.path.join(dst_path, s , "gd_1.008", file_name))
#             data = data[0,:,:]
#             num_frames, num_features = data.shape
#             print(data.shape)
# # Initialize an empty list to store the autocorrelations
#             auto_corrs = []

#             # Iterate through each frame (row)
#             for i in range(num_frames):
#                 new_frame = data[i , :]  # Get the i-th frame (1D array with 512 values)
#                 mean = np.mean(new_frame)
#                 new_data = data- mean
#                 # data_min = np.min(new_frame)
#                 # data_max = np.max(new_frame)
#                 # normalized_data = (new_frame - data_min) / (data_max - data_min + 1e-8)
#                 # corr = signal.correlate(normalized_data, normalized_data,'same' )
#                 # Perform autocorrelation
            
                
#                 # Keep only the second half of the correlation (positive lags)
#                 # auto_corr = auto_corr[len(auto_corr)//2:]
                
#                 # Store the result
#                 auto_corrs.append(new_data)
#             auto_corrs = np.array(auto_corrs)
#             # auto_corrs= np.transpose(auto_corrs)
#             # print(auto_corrs.shape)

# # # Avoid division by zero by adding a small value
# #             normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
#             np.save(os.path.join(dst_path, s ,"gd_thres", file_name), auto_corrs)
            

#     else:
#         for q in noises:
#             os.makedirs(os.path.join(dst_path, s ,q ,"gd_thres"),exist_ok = True)
#             gd = os.listdir(os.path.join(dst_path, s ,q,"gd_1.008"))
#             for file_name in gd:
#                 data = np.load(os.path.join(dst_path, s ,q, "gd_1.008", file_name))
#                 data = data[0,:,:]
                
#                 num_frames, num_features = data.shape
#                 auto_corrs = []

#             # Iterate through each frame (row)
#                 for i in range(num_frames):
#                     new_frame = data[i , :]
#                     mean = np.mean(new_frame)
#                     new_data = data- mean  # Get the i-th frame (1D array with 512 values)
#                     # data_min = np.min(new_frame)
#                     # data_max = np.max(new_frame)
#                     # normalized_data = (new_frame - data_min) / (data_max - data_min + 1e-8)
#                     # corr = signal.correlate(normalized_data, normalized_data,'same' )
#                     # Perform autocorrelation
#                 #     auto_corr = np.correlate(new_frame, new_frame, mode='same')
                    
#                 #     # Keep only the second half of the correlation (positive lags)
#                 #     # auto_corr = auto_corr[len(auto_corr)//2:]
                    
#                 #     # Store the result
#                     auto_corrs.append(new_data)
#                 auto_corrs = np.array(auto_corrs)
#                 # auto_corrs= np.transpose(auto_corrs)
# # # Avoid division by zero by adding a small value
# #             normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
#                 np.save(os.path.join(dst_path, s ,q,"gd_thres", file_name), auto_corrs)
                
                                
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed

# Paths and variables
dst_path = '/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008'
base_path = '/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008'
noises = ['0', '5', '10', '20', 'clean']
splits = ['train', 'test', 'valid']

######## Optimized function ########
def process_file(data_path, save_path, is_test=False):
    """Process a single file for both train/valid and test."""
    data = np.load(data_path)
    data = data[0, :, :]  # Assuming only the first channel is relevant
    num_frames, num_features = data.shape
    
    # Subtract the mean from each frame across the whole array
    mean_values = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mean_values

    # Convert to float if needed
    data_centered = data_centered.astype(np.float32)

    # Apply threshold: values less than 20 are set to 0
    # print("Before thresholding:", data_centered[:5, :5])  # Optional: Inspect first 5 rows, 5 columns
    data_centered[data_centered < 5] = 0
    # print("After thresholding:", data_centered[:5, :5])   # Optional: Check after threshold

    # Directly save the result (all frames processed at once)
    np.save(save_path, data_centered)

######## Process train and valid splits ########
for s in splits:
    if s != "test":
        os.makedirs(os.path.join(dst_path, s, "gd_thres"), exist_ok=True)
        gd_files = os.listdir(os.path.join(dst_path, s, "gd_1.008"))
        
        # Parallelize the processing
        Parallel(n_jobs=-1)(delayed(process_file)(
            os.path.join(dst_path, s, "gd_1.008", file_name),
            os.path.join(dst_path, s, "gd_thres", file_name)
        ) for file_name in tqdm(gd_files))

    else:
        # Process test splits with noise levels
        for q in noises:
            os.makedirs(os.path.join(dst_path, s, q, "gd_thres"), exist_ok=True)
            gd_files = os.listdir(os.path.join(dst_path, s, q, "gd_1.008"))
            
            # Parallelize the processing for test files
            Parallel(n_jobs=-1)(delayed(process_file)(
                os.path.join(dst_path, s, q, "gd_1.008", file_name),
                os.path.join(dst_path, s, q, "gd_thres", file_name)
            ) for file_name in tqdm(gd_files))

# import numpy as np
# import os
# from tqdm import tqdm
# from joblib import Parallel, delayed
# from scipy import signal

# # Paths and variables
# dst_path = '/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008'
# noises = ['0', '5', '10', '20', 'clean']
# splits = ['train', 'test', 'valid']

# ######## Function to normalize, compute autocorrelation, and remove mean ########
# def process_file_with_autocorr(data_path, save_path):
#     """Normalize each frame, compute autocorrelation, and remove mean from autocorrelation."""
#     data = np.load(data_path)
#     data = data[0, :, :]  # Assuming only the first channel is relevant
#     num_frames, num_features = data.shape
    
#     # Pre-allocate space for the autocorrelations
#     auto_corrs = np.zeros((num_frames, 1023))
    
#     # Iterate over each frame
#     for i in range(num_frames):
#         frame = data[i, :]
#         mean = np.mean(frame)
#         std = np.std(frame) + 1e-8  # To prevent division by zero
        
#         # Normalize the frame (subtract mean, divide by std)
#         normalized_frame = (frame - mean) / std
        
#         # Compute the autocorrelation using 'full' mode
#         auto_corr = np.correlate(normalized_frame, normalized_frame, mode='full')
        
#         # Keep the second half (positive lags)
#         # auto_corr = auto_corr[len(auto_corr) // 2:]
        
#         # Truncate the autocorrelation to match the original number of features
#         # auto_corr = auto_corr[:num_features]

#         # Remove the mean from the autocorrelation
#         auto_corr -= np.mean(auto_corr)

#         # Store the result
#         auto_corrs[i, :] = auto_corr

#     # Save the autocorrelation results
#     np.save(save_path, auto_corrs)

# ######## Process train and valid splits ########
# for s in splits:
#     if s != "test":
#         os.makedirs(os.path.join(dst_path, s, "gd_thres"), exist_ok=True)
#         gd_files = os.listdir(os.path.join(dst_path, s, "gd_1.008"))
        
#         # Parallelize the processing
#         Parallel(n_jobs=-1)(delayed(process_file_with_autocorr)(
#             os.path.join(dst_path, s, "gd_1.008", file_name),
#             os.path.join(dst_path, s, "gd_thres", file_name)
#         ) for file_name in tqdm(gd_files))

#     else:
#         # Process test splits with noise levels
#         for q in noises:
#             os.makedirs(os.path.join(dst_path, s, q, "gd_thres"), exist_ok=True)
#             gd_files = os.listdir(os.path.join(dst_path, s, q, "gd_1.008"))
            
#             # Parallelize the processing for test files
#             Parallel(n_jobs=-1)(delayed(process_file_with_autocorr)(
#                 os.path.join(dst_path, s, q, "gd_1.008", file_name),
#                 os.path.join(dst_path, s, q, "gd_thres", file_name)
#             ) for file_name in tqdm(gd_files))
