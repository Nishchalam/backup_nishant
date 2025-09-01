# import numpy as np
# import matplotlib.pyplot as plt
# z = np.load("/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2_mudit_final/train/labels/20_mic_M01_si622.npy")
# # Load the group delay data
# y = np.load("/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2_mudit_final/train/gd_thres/20_mic_M01_si622.npy")

# print(y.shape)
# print(z.shape)
# np.savetxt("ground_truth.txt", z, fmt='%i')
# # Select the group delay for the frame you want to plot
# group_delay_voiced = y[0,328, :]
# group_delay_unvoiced = y[0,593, :]
#   # Replace with the desired row or data for group delay
# print(group_delay.shape)

# # Generate frequency axis: 5000 points evenly spaced between 0 and 4000 Hz
# sampling_rate = 8000  # Sampling rate in Hz (8 kHz)
# nyquist_freq = sampling_rate / 4  # Nyquist frequency (4 kHz)
# freqs_hz = np.linspace(0, nyquist_freq, len(group_delay))



# plt.figure(figsize=(4, 2))  # Smaller figure size
# plt.plot(freqs_hz, group_delay, linewidth=0.8, color='blue')  # Thinner line
# plt.xlabel('Frequency (Hz)', fontsize=8)
# plt.ylabel('RRCGD', fontsize=8)
# plt.title('RRCGD_plot_unvoiced_0_Hz', fontsize=8)
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# # Focus on a meaningful x-axis range
# plt.xlim(0, 2000)  # Reduced range to 0-2000 Hz
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)

# # Tight layout to remove excess white space
# plt.tight_layout()

# # Save the concise plot, trimming extra white space
# plot_path = "/speech/nishanth/clean_research/ptdb_full_data/RRCGD_0_Hz.png"
# plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # Use bbox_inches='tight' to cut white space
# plt.show()
# print(f"Plot saved to: {plot_path}")
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Load the group delay data and ground truth labels
z = np.load("/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2_mudit_final/train/gd_thres/0_mic_M01_si622.npy")
y = np.load("/speech/nishanth/clean_research/ptdb_full_data/new_group_delay_full_1.008_pi_by2_mudit_final/valid/gd_thres/clean_mic_M01_si622.npy")


# Select the group delay for voiced and unvoiced frames
group_delay_voiced = y[0, 328, :]  # Example voiced frame
group_delay_unvoiced = z[0, 328, :]  # Example unvoiced frame

sampling_rate = 8000  # Sampling rate in Hz (8 kHz)
nyquist_freq = sampling_rate / 2  # Nyquist frequency (4 kHz)
freqs_hz = np.linspace(0, nyquist_freq, len(group_delay_voiced))


window_length_ms = 32  # 32 ms
hop_length_ms = 10     # 10 ms
n_fft = 4096          # Zero-padded DFT size
frame_index = 328      # Frame to process
snr_db = 0             # Desired SNR in dB

# Step 1: Load audio file
audio_file = '/speech/nishanth/roots_exps/data/MALE/MIC/M01/mic_M01_si622.wav'  # Replace with your audio file path
audio, sr = librosa.load(audio_file, sr=8000)  # Load with resampling to 8 kHz

# Step 2: Calculate frame and hop lengths in samples
window_length_samples = int(window_length_ms * sr / 1000)
hop_length_samples = int(hop_length_ms * sr / 1000)

# Step 3: Frame the audio using a sliding window
frames = librosa.util.frame(audio, frame_length=window_length_samples, hop_length=hop_length_samples)
print(f"Shape of frames: {frames.shape}")

# Step 4: Extract the 328th frame
if frame_index < frames.shape[1]:
    frame = frames[:, frame_index]  # Shape: (window_length_samples,)
else:
    raise ValueError(f"Frame index {frame_index} exceeds the number of frames {frames.shape[1]}.")

# Step 5: Add White Gaussian Noise at 0 dB SNR
signal_power = np.mean(frame**2)
noise_power = signal_power / (10**(snr_db / 10))
noise = np.sqrt(noise_power) * np.random.randn(len(frame))
noisy_frame = frame + noise

# Step 6: Zero-pad the frame to 512 samples
padded_noisy_frame = np.zeros(4096)
padded_clean_frame = np.zeros(4096)
padded_noisy_frame[:len(noisy_frame)] = noisy_frame
padded_clean_frame[:len(frame)] = frame
dft_clean = np.fft.fft(padded_clean_frame)
# Step 7: Compute the 512-point DFT of the noisy frame
dft_noisy = np.fft.fft(padded_noisy_frame)
magnitude_spectrum_noisy = np.abs(dft_noisy)
magnitude_spectrum_clean = np.abs(dft_clean)
# Step 8: Compute the Group Delay Transform (GDT)
# Group delay: -d(phase)/d(omega)
phase_spectrum = np.angle(dft_noisy)
freq_bins = np.fft.fftfreq(n_fft, d=1/sr)  # Frequency bins
magnitude_spectrum_noisy = 20*np.log10(magnitude_spectrum_noisy + 1e-10) 
magnitude_spectrum_clean = 20*np.log10(magnitude_spectrum_clean + 1e-10) 

# # Step 9: Plot the GDT for positive frequencies
# plt.figure(figsize=(10, 6))
# plt.subplot(2,1,1)
# plt.plot(freq_bins[:n_fft//2], magnitude_spectrum_noisy[:n_fft//2 ])  # Positive frequencies only
# plt.title("0db_frame_mag_plot", fontsize=14)
# plt.xlabel("Frequency (Hz)", fontsize=12)
# plt.ylabel("20log10_mag_dft (s)", fontsize=12)
# plt.grid(alpha=0.5)

# plt.subplot(2,1,2)
# plt.plot(freq_bins[:n_fft//2], magnitude_spectrum_clean[:n_fft//2 ])  # Positive frequencies only
# plt.title("clean_frame_mag_plot", fontsize=14)
# plt.xlabel("Frequency (Hz)", fontsize=12)
# plt.ylabel("20log10_mag_dft (s)", fontsize=12)
# plt.grid(alpha=0.5)
# plt.tight_layout()
# plot_path = "/speech/nishanth/clean_research/ptdb_full_data/0db_frame_328.png"
# plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.show()


# Create a figure with two subplots
plt.figure(figsize=(12, 9))  # Adjust figure size for compact display

# Plot for voiced frame
plt.subplot(4, 1, 1)  # 3 rows, 1 column, first plot
plt.plot(freqs_hz, group_delay_voiced, linewidth=1.7, color='blue')
#  plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
plt.ylabel('RRCGD', fontsize=16, fontweight='bold')
plt.title('RRCGD(clean)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlim(0, 4000)  # Focus on 0-2000 Hz range
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.subplot(4, 1, 2)  # 3 rows, 1 column, third plot
plt.plot(freq_bins[:n_fft // 2], magnitude_spectrum_clean[:n_fft // 2], linewidth=1.7, color='green')
# plt.xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
plt.ylabel('Log Magnitude', fontsize=16, fontweight='bold')
plt.title('DFT spectrum(clean)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlim(0, 4000)
plt.ylim(-40, 19)  # Focus on 0-2000 Hz range
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()
# Plot for unvoiced frame
plt.subplot(4, 1, 3)  # 3 rows, 1 column, second plot
plt.plot(freqs_hz, group_delay_unvoiced, linewidth=1.7, color='blue')
# plt.xlabel(, fontsize=12, fontweight='bold')
plt.ylabel('RRCGD', fontsize=16, fontweight='bold')
plt.title('RRCGD(0dB)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlim(0, 4000)  # Focus on 0-2000 Hz range
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')


# Plot for noisy frame magnitude spectrum
plt.subplot(4, 1, 4)  # 3 rows, 1 column, third plot
plt.plot(freq_bins[:n_fft // 2], magnitude_spectrum_noisy[:n_fft // 2], linewidth=1.7, color='green')
plt.xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
plt.ylabel('Log Magnitude', fontsize=16, fontweight='bold')
plt.title('DFT spectrum(0dB)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlim(0, 4000)
plt.ylim(-20, 20)  # Focus on 0-2000 Hz range
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the plot
plot_path = "/speech/nishanth/clean_research/ptdb_full_data/final_plot_4000hz.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # Use bbox_inches='tight' to trim white space
plt.show()

print(f"Plot saved to: {plot_path}")
