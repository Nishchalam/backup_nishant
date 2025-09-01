import numpy as np
import soundfile as sf
# Example arrays
# audio_path ="/speech/nishanth/root_exps/data/FEMALE/MIC/F01/mic_F01_sa2.wav"
# audio, sample_rate = sf.read(audio_path)
# np.savetxt('audio_data.txt',audio,fmt='%f')
array1 = np.load("/speech/nishanth/raw_exps/full_final_data_8khz_rev/train/labels/0a_mic_F01_sa2.npy")
array2 = np.load("/speech/nishanth/raw_exps/full_final_data_8khz_rev/train/labels/0r_mic_F01_sa2.npy")

print(np.shape(array1))
print(np.shape(array2))
# Stack the arrays side by side (horizontally)
#combined_array = np.hstack((array1.reshape(-1, 1), array2.reshape(-1, 1)))

# Save the combined array to a text file
np.savetxt('output.txt', array1, fmt='%f')
np.savetxt('outputr.txt', array2, fmt='%f')
