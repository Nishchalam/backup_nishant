import os
import argparse
import numpy as np
from tqdm import tqdm

from numpy import angle
from scipy import signal
import multiprocessing as mp



def compute_gd(all_args):

    roots_dir_path, gd_dir, noise_db, split, file_name = all_args
    root_link = os.path.join(roots_dir_path, noise_db, split, file_name)
    gd_save_path = os.path.join(gd_dir, noise_db, split, file_name)

    print(f"processing {file_name}")
    if os.path.exists(gd_save_path):
        print(f"file already exists at {gd_save_path}")

    else:
        radii = [1.002, 1.005, 1.008]
        root_array = np.load(root_link)

        def reflect_roots(root_array, radius):
            indices = np.abs(root_array) > 1
            root_array[indices] = (1 / np.abs(root_array[indices]) ) * (np.exp(complex(0,1)*angle(root_array[indices])) / radius)
            root_array[~indices] = root_array[~indices] / radius

            return root_array
        
        reflected = [reflect_roots(root_array, radius) for radius in radii]

        gd_spectrum = [[], [], []]

        for index in range(len(radii)):
            for frame in reflected[index]:
                gd = [signal.group_delay(([1, -r], [1]), w=1024)[1] for r in frame]
                gd_spectrum[index].append(sum(gd))

        # return np.array(gd_spectrum)[:, :, :512]
        np.save(gd_save_path, np.array(gd_spectrum)[:, :, :512])
        print(f"file saved at {gd_save_path}")


# link = "/home/speech/Desktop/roots/0/train/mic_F06_si1433.npy"
# gd = compute_gd(link)
# print(gd.shape)

def main(args):
    noise_db = str(args.noise_db)
    roots_dir_path = args.roots_dir_path
    gd_dir = args.gd_dir
    nj = int(args.nj)

    pool = mp.Pool(processes=nj)

    os.makedirs(os.path.join(gd_dir, noise_db), exist_ok=True)
    for split in ["train", "test"]:
        print(f"processing {split} split of {noise_db} db .....")
        gd_split_path = os.path.join(gd_dir, noise_db, split)
        if not os.path.exists(gd_split_path):
            os.makedirs(gd_split_path)
        
        to_process = os.listdir(os.path.join(roots_dir_path, noise_db, split))
        to_process = [(roots_dir_path, gd_dir, noise_db, split, file_name) for file_name in to_process]
        
        pool.map(compute_gd, to_process)
        print(f"finished processing {split} split of {noise_db} db .....")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_db", required=True, help="db of noise like 0,5,10,15 etc.")
    parser.add_argument("--roots_dir_path", required=True, help="path to the folder containing a train and a test folder with roots")
    parser.add_argument("--gd_dir", required=True, help="where to save the gd")
    parser.add_argument("--nj", required=True, help="number of jobs")
    args = parser.parse_args()
    main(args)


