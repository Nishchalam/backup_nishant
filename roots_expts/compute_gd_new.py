import os
import argparse
import numpy as np
from tqdm import tqdm
import numba
from numpy import angle
from scipy import signal
import multiprocessing as mp

radii = [1.002]
noise_db_list = ["0","10","20","5","clean"]
splits = ["train", "test", "valid"]
roots_dir_path = "/speech/nishant/roots_exps/roots"
gd_dir = "/speech/nishant/roots_exps/new_group_delay"
nj = 20
pool = mp.Pool(processes=64)

def compute_gd_numba(root_array):

    indices = np.abs(root_array) > 1
    root_array[indices] = (1 / np.abs(root_array[indices]) ) * (np.exp(complex(0,1)*angle(root_array[indices])))
    root_array[~indices] = root_array[~indices]
    gd_spectrum = [np.zeros_like(root_array) for _ in range(len(radii))]
    for index in range(len(radii)): # per radius
        # reflect roots
        r_0 = radii[index]
        for i in range(root_array.shape[0]): # per frame
            for j in range(root_array.shape[1]): # per sample in one frame
                r = root_array[i, j]
                m = np.abs(r)
                theta = np.angle(r)
                w = np.linspace(0, 2*np.pi, 512)
                gd_per_w = np.zeros_like(w)
                for k in range(len(w)): # per w
                    gd = (m**2 - (m*r_0*np.cos(w[k]-theta))) / (r_0**2 + m**2 - 2*m*r_0*np.cos(w[k]-theta))
                    gd_per_w[k] = gd
                gd = np.sum(gd_per_w, axis = 0)
                gd_spectrum[index][i, j] = gd
    gd_spectrum = np.array(gd_spectrum)
    return gd_spectrum

    
    
def reflect_roots(root_array, radius):
    indices = np.abs(root_array) > 1
    root_array[indices] = (1 / np.abs(root_array[indices]) ) * (np.exp(complex(0,1)*angle(root_array[indices])))
    root_array[~indices] = root_array[~indices]
    return root_array
    
    
def compute_gd_numpy(root_path, gd_path):
    orig_root_array = np.load(root_path)
    gd_spectrum = np.array([np.zeros_like(orig_root_array) for _ in range(len(radii))])
    
    root_array = np.array([reflect_roots[orig_root_array, radius]] for radius in radii) # shape : (n_radius, n_frames, n_samples)
    m = np.abs(orig_root_array) # shape : (n_frames,n_samples)
    theta = np.angle(orig_root_array) # shape : (n_frames,n_samples)
    root_array = root_array[:, :, :, np.newaxis] # shape : (n_radius, n_frames, n_samples, 1)
    pass    


def main():
    to_process = []
    for noise_db in noise_db_list:
        for split in splits:
            base_path = os.path.join(roots_dir_path, noise_db, split)
            gd_save_path = os.path.join(gd_dir, noise_db, split)
            if not os.path.exists(gd_save_path):
                os.makedirs(gd_save_path, exist_ok=True)
            roots_path = [os.path.join(base_path, path) for path in os.listdir(base_path)]
            roots_path = [path for path in roots_path if not os.path.exists(os.path.join(gd_save_path, os.path.basename(path)))]
            
            to_process = [(path, os.path.join(gd_save_path, os.path.basename(path))) for path in roots_path]
            for tup in to_process:
                arr = np.load(tup[0])
                gd_spectrum = compute_gd_numba(tup[0])
                np.save(tup[1], np.array(gd_spectrum)[:, :, :512])
            
            

main()