import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf

root_path = "/speech/nishanth/roots_exps/data"
os.makedirs("paths", exist_ok=True)

all_paths = []
for gen in ["MALE", "FEMALE"]:
    if gen == "MALE":
        for ix in range(1,10):
            all_paths.extend(os.path.join(root_path, gen, "MIC", "M0" + str(ix), x) for x in os.listdir(os.path.join(root_path, gen, "MIC", "M0" + str(ix))))
        all_paths.extend(os.path.join(root_path, gen, "MIC", "M10", x) for x in os.listdir(os.path.join(root_path, gen, "MIC", "M10")))
    elif gen == "FEMALE":
        for ix in range(1,10):
            all_paths.extend(os.path.join(root_path, gen, "MIC", "F0" + str(ix), x) for x in os.listdir(os.path.join(root_path, gen, "MIC", "F0" + str(ix))))
        all_paths.extend(os.path.join(root_path, gen, "MIC", "F10", x) for x in os.listdir(os.path.join(root_path, gen, "MIC", "F10")))

print(len(all_paths))

np.random.shuffle(all_paths)

dbs = [0, 5, 10, 20, "clean"]
paths_per_db = np.array_split(all_paths, len(dbs))

for db, db_paths in zip(dbs, paths_per_db):
    train_paths = db_paths[:int(0.8 * len(db_paths))]
    test_paths = db_paths[int(0.8 * len(db_paths)) + 1:]
    
    os.makedirs(os.path.join("paths", str(db)), exist_ok=True)
    
    with open(os.path.join("paths", str(db), "train.scp"), "w") as f:
        for path_name in train_paths:
            f.write(f"{path_name}\n")
            
    with open(os.path.join("paths", str(db), "test.scp"), "w") as f:
        for path_name in test_paths:
            f.write(f"{path_name}\n")
