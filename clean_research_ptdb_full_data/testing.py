import numpy as np
import soundfile as sf

# def AWGN(audio, SNR_dB):

#     p0 = 1e-5

#     audio_avg_power = np.mean(audio ** 2)
#     audio_avg_power_dB = 10 * np.log10(audio_avg_power / p0)

#     noise_avg_dB = audio_avg_power_dB - SNR_dB
#     noise_avg_power = (10 ** (noise_avg_dB / 10)) * p0

#     noisy_signal = np.random.normal(0, np.sqrt(noise_avg_power), len(audio))

#     return noisy_signal + audio

# # def add_noise(signal, snr_dB):
# #     # Calculate signal power and convert to dB
# #     signal_power = np.mean(signal**2)
# #     signal_power_dB = 10 * np.log10(signal_power)
    
# #     # Calculate noise power in dB
# #     noise_power_dB = signal_power_dB - snr_dB
    
# #     # Convert noise power to linear scale
# #     noise_power = 10**(noise_power_dB / 10)
    
# #     # Generate white Gaussian noise
# #     noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    
# #     # Add noise to the signal
# #     noisy_signal = signal + noise
# #     return noisy_signal

# def add_noise_to_wav(input_wav, output_wav, snr_dB):
#     # Load the input WAV file
#     audio, sample_rate = sf.read(input_wav)
    
#     # Add noise
#     noisy_signal = AWGN(audio, snr_dB)
    
#     # Save the noisy signal to the output WAV file
#     sf.write(output_wav, noisy_signal, sample_rate)

# # Example usage
# input_wav = '/speech/nishant/roots_exps/data/FEMALE/MIC/F01/mic_F01_sa1.wav'
# output_wav_0db1 = 'output_0db1.wav'
# output_wav_10db1 = 'output_10db1.wav'
# output_wav_20db1 = 'output_20db1.wav'
# output_wav_5db1 = 'output_5db1.wav'
# # Add noise at 10 dB SNR
# add_noise_to_wav(input_wav, output_wav_10db1, 10)

# # Add noise at 20 dB SNR

# add_noise_to_wav(input_wav, output_wav_0db1, 0)
# add_noise_to_wav(input_wav, output_wav_5db1, 5)
# add_noise_to_wav(input_wav, output_wav_20db1, 20)


##############################################################################################################################################
# norm =np.load("/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008/train/labels/0_mic_F01_sa2.npy")
# # norm = norm[0,:,:]
# print(norm.shape)
# # print(np.mean(norm))
# # actual = np.load("/speech/nishanth/clean_research/ptdb_full_data/mudit_full_final_data_1.008/train/gd_thres/0_mic_F01_sa1.npy")
# # actual = actual[0,:,:]
# # print(np.mean(actual))
# np.savetxt('norm.txt', norm, fmt='%2f')
# # np.savetxt('actual.txt', actual, fmt='%2f')
import os
import numpy as np
from numpy import angle
from scipy import signal

# Define the output path
out_path = "/speech/nishanth/clean_research/ptdb_full_data/touch.npy"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# Load the input numpy file
input_path = "/speech/nishanth/clean_research/ptdb_full_data/roots/clean/train/mic_F01_sa2.npy"

# Check if the file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"The input file {input_path} does not exist.")

# Load the root array
root_array = np.load(input_path)
print("Roots array shape: ", root_array.shape)

# Process the root array
indices = np.where(np.abs(root_array) > 1)
root_array[indices] = (1 / np.abs(root_array[indices])) * (np.exp(complex(0,1) * angle(root_array[indices])))

# Parameters
radii = [1.008]
gd_n_points = 5000

# Initialize the gd_spectrum
gd_spectrum = [np.zeros((root_array.shape[0], root_array.shape[1], gd_n_points)) for _ in range(len(radii))]

# Main processing loop
for index in range(len(radii)):  # per radius
    r_0 = radii[index]
    for i in range(root_array.shape[0]):  # per frame
        for j in range(root_array.shape[1]):  # per sample in one frame
            r = root_array[i, j]
            m = np.abs(r)
            theta = np.angle(r)
            w = np.linspace(0, np.pi, gd_n_points)
            gd_per_w = np.zeros_like(w)
            for k in range(len(w)):  # per w
                gd = (m**2 - (m * r_0 * np.cos(w[k] - theta))) / (r_0**2 + m**2 - 2 * m * r_0 * np.cos(w[k] - theta))
                gd_per_w[k] = gd
            gd_spectrum[index][i, j] = np.array(gd_per_w)

# Summing the gd_spectrum across the third axis
gd_spectrum = np.array(gd_spectrum)
gd_spectrum = np.sum(gd_spectrum, axis=2)
gd_spectrum = gd_spectrum[0, :, :]

# Check the shape of the result
print("GD Spectrum shape:", gd_spectrum.shape)

# Save the output
np.save(out_path, np.array(gd_spectrum)[:, :5000])

# Confirm file saved
if os.path.exists(out_path):
    print(f"File successfully saved at {out_path}")
else:
    print("File saving failed.")

