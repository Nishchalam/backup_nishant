import numpy as np
import matplotlib.pyplot as plt
from numpy import angle

# Generate the signal x(n) = sin(n * pi / 4) for n = 0 to 79
n = np.arange(80)
x = np.sin(n * np.pi / 4)
# x = x[:32]
# Compute the roots of the signal
roots = np.roots(np.flip(x))

# Reflect roots outside the unit circle
indices = np.where(np.abs(roots) > 1)
roots[indices] = (1 / np.abs(roots[indices])) * (np.exp(1j * angle(roots[indices])))

# Initialize RRCGD spectrum array
gd_n_points = 1024
gd_spectrum = np.zeros((roots.shape[0], gd_n_points))

# Parameters for computation
r_0 = 1.008
w = np.linspace(0, np.pi , gd_n_points)

# Compute the group delay spectrum for each root
for j in range(roots.shape[0]):  # Per root
    r = roots[j]
    m = np.abs(r)
    theta = np.angle(r)
    gd_spectrum[j, :] = (m**2 - (m * r_0 * np.cos(w - theta))) / (
        r_0**2 + m**2 - 2 * m * r_0 * np.cos(w - theta)
    )

# Sum the group delay spectrum across all roots
gd_spectrum = np.sum(gd_spectrum, axis=0)
print(gd_spectrum.shape)
gd_spectrum = gd_spectrum[:512]
print(gd_spectrum.shape)
# Define the frequency axis
sampling_rate = 8000  # Sampling rate in Hz (8 kHz)
nyquist_freq = sampling_rate / 2 # Nyquist frequency (4 kHz)
freqs_hz = np.linspace(0, nyquist_freq, len(gd_spectrum))

# Plot the RRCGD spectrum
plt.figure(figsize=(15,8))  # Smaller figure size
plt.plot(freqs_hz, gd_spectrum, linewidth=0.8, color='blue')  # Thinner line
plt.xlabel('Frequency (Hz)', fontsize=8)
plt.ylabel('RRCGD', fontsize=8)
# plt.title('RRCGD_plot', fontsize=8)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Focus on a meaningful x-axis range
# plt.xlim(500, 1500)  # Focus on 500 Hz to 1500 Hz
plt.ylim(np.min(gd_spectrum) +40, np.max(gd_spectrum) + 5) # Reduced range to 0-2000 Hz
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

# Tight layout to remove excess white space
plt.tight_layout()

# Save the concise plot, trimming extra white space
plot_path = "/speech/nishanth/clean_research/ptdb_full_data/RRCGD_1.008.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # Use bbox_inches='tight' to cut white space
plt.show()
