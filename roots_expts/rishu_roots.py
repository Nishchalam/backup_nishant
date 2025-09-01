import os

import librosa

import numpy as np

from tqdm import tqdm
import soundfile as sf
import multiprocessing as mp


# wavs = open("/speech/nishanth/roots_exps/paths/0/test.scp", "r").read().splitlines()
# wavs = [x.strip().split(maxsplit=1)[0] for x in wavs][0]

"""                  Breaking it down into two stages : 
    
    1. FRAME PREPARATION
    
       Generate frame array from audio link.  This includes pre-emphasis and AWGN.
       Output is of shape (N_Frames, Frame_Length).
       e.g. if using 25ms window with hop length 10ms at 16khz sampling rate for a 
            1sec audio clip :- 

            hop_length = 0.010 * 16000 = 160 samples
            frame_length = 0.025 * 16000 = 400 samples
            
            n_audio_samples = 1 * 16000 = 16000 samples

            Thus, the ouput will have shape :-
            n_output_frames = n_audio_samples / hop_length  (audio samples are zero padded)

            output_frame_length = frame_length - 1 (Due to differencing during pre-emphasis)         
    
    
    2. ROOT FINDING

        At the moment can't find a better(faster) way than numpy.roots.
        For each frame in the frame array, compute the roots and store in root array.

        Thus, the output is an array of shape :- 
        (n_output_frames, output_frame_length - 1) 
        The minus 1 for output_frame_length is because a n sample input is considered to be the coefficient array of an n-1 degree polynomial. 
        Thus, numpy.roots results in an array of length n-1.

"""

""" Define Variables """

SR = 8000 #Sampling Rate

hop_length = 80 #hop-length in ms * SR = 10ms * 16000
frame_length = 256 #frame-length in ms * SR = 25ms * 16000

# noise_dB = 0 #Desired noise level in dB.

def AWGN(audio, SNR_dB):

    p0 = 1e-5

    audio_avg_power = np.mean(audio ** 2)
    audio_avg_power_dB = 10 * np.log10(audio_avg_power / p0)

    noise_avg_dB = audio_avg_power_dB - SNR_dB
    noise_avg_power = (10 ** (noise_avg_dB / 10)) * p0

    awgn = np.random.normal(0, np.sqrt(noise_avg_power), len(audio))

    return audio + awgn

def frame_and_compute_roots(args):
    link, noise_level, roots_save_path = args
    print(f"processing {os.path.basename(link)}")
    if os.path.exists(os.path.join(roots_save_path, os.path.basename(link).replace(".wav", ".npy"))):
        print(f"{os.path.basename(link).replace('.wav', '.npy')} already exists")
    else:
        audio, sr = sf.read(link)
        assert len(audio.shape) < 3

        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # print("Audio Time : ", round(len(audio) / sr, 2))
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        # audio = audio[:-1] - 0.98 * audio[1:] #Pre-emphasis with alpha=0.98

        if noise_level!="clean":
            # print(f"Adding noise to make SNR - - > {noise_level}dB.")
            audio = AWGN(audio, SNR_dB=int(noise_level))
            # sf.write("rishu_noise.wav", audio, SR)

        audio_samples = len(audio)
        n_frames = (audio_samples // hop_length) - 2
        # print(f"n_frames : {n_frames}")

        if (audio_samples % hop_length) <= (frame_length % hop_length):
            audio = audio[: (n_frames - 1) * hop_length + frame_length]
            n_frames -= 1
        else:
            audio = audio[: n_frames * hop_length + frame_length]

        frames = [audio[(i*hop_length):(i*hop_length)+frame_length] for i in range(n_frames)]
        
        roots = [np.roots(x[::-1]) for x in frames]
        np.save(os.path.join(roots_save_path, os.path.basename(link).replace(".wav", ".npy")), roots)
        print(f"processing complete for {os.path.basename(link).replace('.wav', '.npy')}")

# main code
os.makedirs("roots", exist_ok=True)
paths_dir = "/speech/nishanth/roots_exps/paths"
roots_save_dir = "/speech/nishanth/roots_exps/roots"
os.makedirs(roots_save_dir, exist_ok=True)
pool = mp.Pool(processes=16)


splits = ["train", "test"]
# for db in os.listdir(paths_dir):
# db = "5"
for db in ["clean"]:
    for split in splits:
        roots_save_path = os.path.join(roots_save_dir, db, split)
        to_process = []
        os.makedirs(roots_save_path, exist_ok=True)
        wavs = open(os.path.join(paths_dir, db, split + ".scp"), "r").read().splitlines()
        
        print(f"processing {split} split of {db} db ...\nroots are saved at {roots_save_path}")
        
        to_process.extend((link, db, roots_save_path) for link in wavs)
        # list(tqdm(pool.imap_unordered(frame_and_compute_roots, to_process), total = len(to_process)))
        pool.map(frame_and_compute_roots, to_process)
        
        print(f"finished processing {split} split of {db} db")
    
        
