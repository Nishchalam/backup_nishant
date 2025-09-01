import numpy as np

# Reference frequency (A1 = 55 Hz)
f0 = 10

# Load the data (assuming space-separated columns: ground truth and predicted pitch in cents)
data = np.loadtxt('/speech/nishanth/clean_research/ptdb_full_data/new_results.txt')

# Separate ground truth (GT) and predicted (Pred) values in cents
ground_truth_cents = data[:, 0]
predicted_cents = data[:, 1]

# Convert GT and Pred from cents to Hz using the formula: f_Hz = f0 * 2^(cents / 1200)
def cents_to_hz(cents, f0):
    return f0 * 2**(cents / 1200)

# Define loose thresholds for halving and doubling detection
halving_threshold = 0.3  # 10% threshold for halving (loose)
doubling_threshold = 0.3  # 10% threshold for doubling (loose)

# Initialize counters for pitch halving and doubling
pitch_halving_count = 0
pitch_doubling_count = 0

# Prepare an empty list to store the results
results = []

# Loop through the data and count halving and doubling errors for voiced pitch (GT > 0)
for gt_cents, pred_cents in zip(ground_truth_cents, predicted_cents):
# if gt_cents > 0:  # Only consider voiced pitch (i.e., non-zero ground truth in cents)
    gt_hz = cents_to_hz(gt_cents, f0)
    pred_hz = cents_to_hz(pred_cents, f0)

    halving = False
    doubling = False

    # Check for pitch halving
    if abs(pred_hz - 0.5 * gt_hz) / (0.5 * gt_hz) < halving_threshold:
        pitch_halving_count += 1
        halving = True
    
    # Check for pitch doubling
    elif abs(pred_hz - 2 * gt_hz) / (2 * gt_hz) < doubling_threshold:
        pitch_doubling_count += 1
        doubling = True

    # Store the result for this frame: ground truth Hz, predicted Hz, halving, doubling
    results.append([gt_hz, pred_hz, halving, doubling])

# Save the results to a new text file
with open('/speech/nishanth/clean_research/ptdb_full_data/halving_doubling_results.txt', 'w') as f:
    f.write("Ground_Truth_Hz\tPredicted_Hz\tPitch_Halving\tPitch_Doubling\n")
    for result in results:
        f.write(f"{result[0]:.2f}\t{result[1]:.2f}\t{result[2]}\t{result[3]}\n")

# Output the summary results
print(f'Number of pitch halving errors: {pitch_halving_count}')
print(f'Number of pitch doubling errors: {pitch_doubling_count}')
