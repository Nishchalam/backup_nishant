import os

import librosa

import numpy as np
from scipy.linalg import toeplitz
import cvxpy as cvx

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

hop_length = 80 #hop-length in ms * SR = 10ms * 8000
frame_length = 256  #frame-length in ms * SR = 32ms * 8000

# noise_dB = 0 #Desired noise level in dB.

def AWGN(audio, SNR_dB):

    p0 = 1e-5

    audio_avg_power = np.mean(audio ** 2)
    audio_avg_power_dB = 10 * np.log10(audio_avg_power / p0)

    noise_avg_dB = audio_avg_power_dB - SNR_dB
    noise_avg_power = (10 ** (noise_avg_dB / 10)) * p0

    awgn = np.random.normal(0, np.sqrt(noise_avg_power), len(audio))

    return audio + awgn
def magnitude_spectrum(frame, n_fft=1024):
    # Compute the FFT of the frame
    fft_output = np.fft.rfft(frame, n=n_fft)
    # Compute the magnitude spectrum and take only the first 512 points
    mag = np.abs(fft_output)[:512]
    return mag

def frame(args):
    
    link, noise_level, frames_save_path = args
    print(f"Processing {os.path.basename(link)}")
    if os.path.exists(os.path.join(frames_save_path, os.path.basename(link).replace(".wav", ".npy"))):
        print(f"{os.path.basename(link).replace('.wav', '.npy')} already exists")
    else:
        audio, sr = sf.read(link)
        assert len(audio.shape) < 3

        if len(audio.shape) > 1:
            audio = audio[:, 0]
            
        # audio = audio[::-1]
        
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        if noise_level != "clean":
            audio = AWGN(audio, SNR_dB=int(noise_level))
        
        audio_samples = len(audio)
        n_frames = (audio_samples // hop_length) - 2

        if (audio_samples % hop_length) <= (frame_length % hop_length):
            audio = audio[: (n_frames - 1) * hop_length + frame_length]
            n_frames -= 1
        else:
            audio = audio[: n_frames * hop_length + frame_length]
        frames_with_mag = []
        for i in range(n_frames):
            frame = audio[(i * hop_length):(i * hop_length) + frame_length]
            
            # If the frame is shorter than frame_length, pad it
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant', constant_values=0)
            
            # Compute the magnitude spectrum and take only the first 512 points
            mag = magnitude_spectrum(frame, n_fft=1024)
            
            frames_with_mag.append(mag)
        # frames = []
        # for i in range(n_frames):
        #     frame = audio[(i * hop_length):(i * hop_length) + frame_length]
            
        #     # If the frame is shorter than frame_length, pad it
        #     if len(frame) < frame_length:
        #         frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant', constant_values=0)
            
        #     frames.append(frame)
        
        # Save the padded frames as a .npy file
        np.save(os.path.join(frames_save_path, os.path.basename(link).replace(".wav", ".npy")), frames_with_mag)
        print(f"Processing complete for {os.path.basename(link).replace('.wav', '.npy')}")


# main code
paths_dir = "/speech/nishanth/raw_exps/paths_full"
frames_save_dir = "/speech/nishanth/raw_exps/raw_full_frames_8Khz"
os.makedirs(frames_save_dir, exist_ok=True)
pool = mp.Pool(processes=20)


splits = ["train", "test","valid"]
# for db in os.listdir(paths_dir):
# db = "5"
for db in [0,5,10,20,"clean"]:
    for split in splits:
        frames_save_path = os.path.join(frames_save_dir, str(db), split)
        to_process = []
        os.makedirs(frames_save_path, exist_ok=True)
        wavs = open(os.path.join(paths_dir, str(db), split + ".scp"), "r").read().splitlines()
        
        print(f"processing {split} split of {db} db ...\nframes are saved at {frames_save_path}")
        
        to_process.extend((link, db, frames_save_path) for link in wavs)
        # list(tqdm(pool.imap_unordered(frame_and_compute_roots, to_process), total = len(to_process)))
        pool.map(frame, to_process)
        
        print(f"finished processing {split} split of {db} db")
    