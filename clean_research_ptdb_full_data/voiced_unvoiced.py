import librosa
import numpy as np

# Step 1: Load and Resample the Audio to 8kHz
audio_path = '/speech/nishanth/roots_exps/data/MALE/MIC/M01/mic_M01_sa1.wav'
audio, sr = librosa.load(audio_path, sr=8000)  # Resample to 8kHz

# Step 2: Define Parameters for Framing
frame_length = int(0.032 * sr)  # 32 ms frame length (in samples)
hop_length = int(0.01 * sr)     # 10 ms hop length (in samples)

# Step 3: Compute Short-Time Energy or ZCR for Voiced/Unvoiced Detection
# Here using short-time energy for simplicity
energy = np.array([
    np.sum(np.abs(audio[i:i + frame_length]**2))
    for i in range(0, len(audio), hop_length)
])

# Normalize energy between 0 and 1
energy = energy / np.max(energy)

# Set threshold for voiced/unvoiced detection (you can fine-tune this threshold)
energy_threshold = 0.1

# Step 4: Create Array: 0 for Unvoiced, 100 for Voiced
voiced_unvoiced = np.where(energy > energy_threshold, 100, 0)

# Step 5: Save the Resulting Array
np.save('voiced_unvoiced_array.npy', voiced_unvoiced)
print(np.shape(voiced_unvoiced))

print("Voiced/Unvoiced array saved successfully!")
/speech/nishanth/clean_research/ptdb_full_data/full_final_data/train/labels/5_mic_M01_sa1.npy