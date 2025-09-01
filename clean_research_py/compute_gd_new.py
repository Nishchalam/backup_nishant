import os
import argparse
from functools import partial
import numpy as np
from tqdm import tqdm
from numpy import angle
from scipy import signal
import multiprocessing as mp

"""
radii = [1.002,1.005,1.008]
noise_db_list = ["0","10","20","5","clean"]
splits = ["train", "test"]
roots_dir_path = "/speech/nishant/clean_research/roots"
gd_dir = "/speech/nishant/clean_research/new_group_delay"
nj = 4
pool = mp.Pool(processes=nj)


def compute_gd(tup):
    print(tup)
    try:
        root_array = np.load(tup[0])

        indices = np.where(np.abs(root_array) > 1)
        root_array[indices] = (1 / np.abs(root_array[indices]) ) * (np.exp(complex(0,1)*angle(root_array[indices])))
        # root_array[~indices] = root_array[~indices]
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
        np.save(tup[1], np.array(gd_spectrum)[:, :, :512])
    except:
        print("failed to process", tup[0])

    
    

    

def main():
    to_process = []
    for noise_db in noise_db_list:
        for split in splits:
            base_path = os.path.join(roots_dir_path, noise_db, split)
            gd_save_path = os.path.join(gd_dir, noise_db, split)
            if not os.path.exists(base_path):
                print(f"Directory '{base_path}' does not exist.")
                continue  # Skip processing this noise_db/split combination
            if not os.path.exists(gd_save_path):
                os.makedirs(gd_save_path, exist_ok=True)
            roots_path = [os.path.join(base_path, path) for path in os.listdir(base_path)]
            roots_path = [path for path in roots_path if not os.path.exists(os.path.join(gd_save_path, os.path.basename(path)))]
            
            to_process = [(path, os.path.join(gd_save_path, os.path.basename(path))) for path in roots_path]
            pool.map(compute_gd, to_process)
            # for tup in to_process:
            #     try:
            #         arr = np.load(tup[0])
            #         gd_spectrum = compute_gd(arr)  # Pass the loaded array to compute_gd
            #         np.save(tup[1], np.array(gd_spectrum)[:, :, :512])
            #     except Exception as e:
            #         print(f"Error processing {tup[0]}: {e}")
            
if __name__ == "__main__":
    main()
"""

def compute_gd(tup, radii):
    gd_n_points = 512
    in_path, out_path = tup

    try:
        print("Processing:", in_path)
        root_array = np.load(in_path)
        # print("roots shape : ", root_array.shape)

        indices = np.where(np.abs(root_array) > 1)
        root_array[indices] = (1 / np.abs(root_array[indices]) ) * (np.exp(complex(0,1)*angle(root_array[indices])))
        # root_array[~indices] = root_array[~indices]
        # gd_spectrum = [np.zeros_like(root_array) for _ in range(len(radii))]
        gd_spectrum = [np.zeros((root_array.shape[0], root_array.shape[1], gd_n_points)) for _ in range(len(radii))]

        for index in range(len(radii)): # per radius
            # reflect roots
            r_0 = radii[index]
            for i in range(root_array.shape[0]): # per frame
                for j in range(root_array.shape[1]): # per sample in one frame
                    r = root_array[i, j]
                    m = np.abs(r)
                    theta = np.angle(r)
                    w = np.linspace(0, (np.pi)*0.5, gd_n_points)
                    gd_per_w = np.zeros_like(w)
                    for k in range(len(w)): # per w
                        gd = (m**2 - (m*r_0*np.cos(w[k]-theta))) / (r_0**2 + m**2 - 2*m*r_0*np.cos(w[k]-theta))
                        gd_per_w[k] = gd
                    gd_spectrum[index][i, j] = np.array(gd_per_w)
        gd_spectrum = np.array(gd_spectrum)
        gd_spectrum = np.sum(gd_spectrum, axis = 2)
        print(gd_spectrum.shape)
        np.save(out_path, np.array(gd_spectrum)[:, :, :512])# Your compute_gd function code here
    except Exception as e:
        print("Error processing", tup[0], ":", e)

def load_data(path):
    try:
        return np.load(path)
    except Exception as e:
        print("Error loading data from", path, ":", e)
        return None

def process_data(data, radii):
    try:
        # Your data processing code here
        pass
    except Exception as e:
        print("Error processing data:", e)

def main():
    radii = [1.008]
    noise_db_list = ["0", "clean","5","10","20"]
    splits = ["train", "test", "valid"]
    roots_dir_path = "/speech/nishanth/clean_research/ptdb_full_data/roots"
    gd_dir = "/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2"
    nj = 20
    pool = mp.Pool(processes=nj)

    to_process = []
    for noise_db in noise_db_list:
        for split in splits:
            base_path = os.path.join(roots_dir_path, noise_db, split)
            gd_save_path = os.path.join(gd_dir, noise_db, split)
            if not os.path.exists(base_path):
                print(f"Directory '{base_path}' does not exist.")
                continue
            if not os.path.exists(gd_save_path):
                os.makedirs(gd_save_path, exist_ok=True)
            roots_path = [os.path.join(base_path, path) for path in os.listdir(base_path)]
            roots_path = [path for path in roots_path if not os.path.exists(os.path.join(gd_save_path, os.path.basename(path)))]
            to_process.extend([(path, os.path.join(gd_save_path, os.path.basename(path))) for path in roots_path])
    
    compute_gd_partial = partial(compute_gd, radii=radii)
    pool.map(compute_gd_partial, to_process)

if __name__ == "__main__":
    main()
