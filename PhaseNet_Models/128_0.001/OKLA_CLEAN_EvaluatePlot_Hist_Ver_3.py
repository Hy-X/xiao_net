#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import matplotlib.pyplot as plt
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
import numpy as np
from scipy.signal import find_peaks

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model and move it to the device
model = sbm.PhaseNet(phases="PSN")
model.to(device)

# Load the data
data = sbd.OKLA_CLEAN(sampling_rate=100, force=True)
train, dev, test = data.train_dev_test()

# Set up data augmentation
phase_dict = {"p_arrival_sample": "P", "s_arrival_sample": "S"}
train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)

augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
    sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(sigma=30, dim=0)
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)

# Load the model
def load_model(model, model_filename, device):
    """
    Load the model from a file.
    Args:
        model: Model architecture to load the weights into.
        model_filename: Filename of the saved model.
        device: Device to load the model onto.
    Returns:
        Model with loaded weights.
    """
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.to(device)
    return model

# Load the final model
model_filename = "final_model.pth"
model = load_model(model, model_filename, device)

# Create the output folder
output_folder = "single_examples"
os.makedirs(output_folder, exist_ok=True)

# Parameters for peak detection
sampling_rate = 100  # Samples per second (100 Hz)
height = 0.5
distance = 100

# Run the prediction 100 times
for i in range(1, 101):
    # Visualizing Predictions
    sample = dev_generator[np.random.randint(len(dev_generator))]

    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]})
    axs[0].plot(sample["X"].T)
    axs[1].plot(sample["y"].T)

    # Find peaks in the ground truth labels
    y_p_peaks, _ = find_peaks(sample["y"][0], height=height, distance=distance)
    y_s_peaks, _ = find_peaks(sample["y"][1], height=height, distance=distance)

    # Convert ground truth peak indices to time values
    y_p_arrival_times = y_p_peaks / sampling_rate
    y_s_arrival_times = y_s_peaks / sampling_rate

    axs[1].plot(y_p_peaks, sample["y"][0, y_p_peaks], 'o', label='P arrivals', color='red')
    axs[1].plot(y_s_peaks, sample["y"][1, y_s_peaks], 'o', label='S arrivals', color='blue')
    axs[1].legend()

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        pred = model(torch.tensor(sample["X"], device=device).unsqueeze(0))  # Add a fake batch dimension
        pred = pred[0].cpu().numpy()

    # Extract the probability distributions for P and S phases
    p_prob = pred[0]
    s_prob = pred[1]

    # Identify peaks in the probability distributions
    p_peaks, _ = find_peaks(p_prob, height=height, distance=distance)
    s_peaks, _ = find_peaks(s_prob, height=height, distance=distance)

    # Convert peak indices to time values
    p_arrival_times = p_peaks / sampling_rate
    s_arrival_times = s_peaks / sampling_rate

    # Calculate residuals
    residual_p_arrival_times = p_arrival_times - y_p_arrival_times[:, np.newaxis]
    residual_s_arrival_times = s_arrival_times - y_s_arrival_times[:, np.newaxis]

    # Plot the probability distributions and the detected peaks
    axs[2].plot(p_prob, label='P-phase Probability')
    axs[2].plot(p_peaks, p_prob[p_peaks], 'x', label='Detected P Peaks')
    axs[2].plot(s_prob, label='S-phase Probability')
    axs[2].plot(s_peaks, s_prob[s_peaks], 'x', label='Detected S Peaks')
    axs[2].set_title('P-phase and S-phase Probability Distributions')
    axs[2].set_xlabel('Sample Index')
    axs[2].set_ylabel('Probability')
    axs[2].legend()

    plt.tight_layout()
    plot_filename = os.path.join(output_folder, f"Prediction{i}_Predictions_Plot.png")
    plt.savefig(plot_filename)
    #plt.show()
    plt.close(fig)

    # Save the results to a text file
    results_filename = os.path.join(output_folder, f"Prediction{i}_Example_Results.txt")
    with open(results_filename, "w") as f:
        f.write(f"Ground Truth P arrival times: {y_p_arrival_times}\n")
        f.write(f"Ground Truth S arrival times: {y_s_arrival_times}\n")
        f.write(f"Predicted P arrival times: {p_arrival_times}\n")
        f.write(f"Predicted S arrival times: {s_arrival_times}\n")
        f.write(f"Residual P arrival times: {residual_p_arrival_times}\n")
        f.write(f"Residual S arrival times: {residual_s_arrival_times}\n")

    # Save the parameters to a text file
    parameters_filename = os.path.join(output_folder, f"Prediction{i}_Example_Parameters.txt")
    with open(parameters_filename, "w") as f:
        f.write(f"Sampling Rate: {sampling_rate}\n")
        f.write(f"Height Parameter: {height}\n")
        f.write(f"Distance Parameter: {distance}\n")


# In[ ]:


import torch
import matplotlib.pyplot as plt
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
import numpy as np
from scipy.signal import find_peaks

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model and move it to the device
model = sbm.PhaseNet(phases="PSN")
model.to(device)

# Load the data
data = sbd.OKLA_CLEAN(sampling_rate=100, force=True)
train, dev, test = data.train_dev_test()

# Set up data augmentation
phase_dict = {"p_arrival_sample": "P", "s_arrival_sample": "S"}
train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)

augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
    sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(sigma=30, dim=0)
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)

# Load the model
def load_model(model, model_filename, device):
    """
    Load the model from a file.
    Args:
        model: Model architecture to load the weights into.
        model_filename: Filename of the saved model.
        device: Device to load the model onto.
    Returns:
        Model with loaded weights.
    """
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.to(device)
    return model

# Load the final model
model_filename = "final_model.pth"
model = load_model(model, model_filename, device)

# Define parameters for find_peaks
sampling_rate = 100  # Samples per second (100 Hz)
height = 0.5
distance = 100

# Save find_peaks parameters to a file
params = (
    f"Sampling Rate: {sampling_rate} Hz\n"
    f"Height: {height}\n"
    f"Distance: {distance}\n"
)

with open("Find_Peaks_Histogram_Parameters.txt", "w") as file:
    file.write(params)

# Process samples
n_samples = len(dev_generator)

all_residual_p_arrival_times = []
all_residual_s_arrival_times = []

for i in range(n_samples):
    sample = dev_generator[np.random.randint(len(dev_generator))]

    # Find peaks in the ground truth labels
    y_p_peaks, _ = find_peaks(sample["y"][0], height=height, distance=distance)
    y_s_peaks, _ = find_peaks(sample["y"][1], height=height, distance=distance)

    # Convert ground truth peak indices to time values
    y_p_arrival_times = y_p_peaks / sampling_rate
    y_s_arrival_times = y_s_peaks / sampling_rate

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        pred = model(torch.tensor(sample["X"], device=device).unsqueeze(0))  # Add a fake batch dimension
        pred = pred[0].cpu().numpy()

    # Extract the probability distributions for P and S phases
    p_prob = pred[0]
    s_prob = pred[1]

    # Identify peaks in the probability distributions
    p_peaks, _ = find_peaks(p_prob, height=height, distance=distance)
    s_peaks, _ = find_peaks(s_prob, height=height, distance=distance)

    # Convert peak indices to time values
    p_arrival_times = p_peaks / sampling_rate
    s_arrival_times = s_peaks / sampling_rate

    # Calculate residuals for P and S peaks
    for y_p_time in y_p_arrival_times:
        residual_p_arrival_times = p_arrival_times - y_p_time
        all_residual_p_arrival_times.extend(residual_p_arrival_times)
        
    for y_s_time in y_s_arrival_times:
        residual_s_arrival_times = s_arrival_times - y_s_time
        all_residual_s_arrival_times.extend(residual_s_arrival_times)

# Calculate mean and standard deviation for residual P and S arrival times
mean_residual_p_arrival_times = np.mean(all_residual_p_arrival_times)
std_residual_p_arrival_times = np.std(all_residual_p_arrival_times)

mean_residual_s_arrival_times = np.mean(all_residual_s_arrival_times)
std_residual_s_arrival_times = np.std(all_residual_s_arrival_times)

output = (
    f"Mean Residual P Arrival Time: {mean_residual_p_arrival_times:.4f} seconds\n"
    f"Standard Deviation of Residual P Arrival Time: {std_residual_p_arrival_times:.4f} seconds\n"
    f"Mean Residual S Arrival Time: {mean_residual_s_arrival_times:.4f} seconds\n"
    f"Standard Deviation of Residual S Arrival Time: {std_residual_s_arrival_times:.4f} seconds\n"
)

with open("Model_Prediction_Statistics_Output.txt", "w") as file:
    file.write(output)

print(output)

# Plot the histogram of residual P peak arrival times
plt.figure(figsize=(10, 5))
plt.hist(all_residual_p_arrival_times, bins=30, color='skyblue', edgecolor='black', range=(-1, 1))
plt.title('Histogram of Residual P Peak Arrival Times')
plt.xlabel('Residual P Arrival Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('histogram_residual_p_arrival_times.png')
#plt.show()

# Plot the histogram of residual S peak arrival times
plt.figure(figsize=(10, 5))
plt.hist(all_residual_s_arrival_times, bins=30, color='salmon', edgecolor='black', range=(-1, 1))
plt.title('Histogram of Residual S Peak Arrival Times')
plt.xlabel('Residual S Arrival Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('histogram_residual_s_arrival_times.png')
#plt.show()


# In[ ]:


import torch
import matplotlib.pyplot as plt
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
import numpy as np
from scipy.signal import find_peaks

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model and move it to the device
model = sbm.PhaseNet(phases="PSN")
model.to(device)

# Load the data
data = sbd.OKLA_CLEAN(sampling_rate=100, force=True)
train, dev, test = data.train_dev_test()

# Set up data augmentation
phase_dict = {"p_arrival_sample": "P", "s_arrival_sample": "S"}
train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)

augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
    sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(sigma=30, dim=0)
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)

# Load the model
def load_model(model, model_filename, device):
    """
    Load the model from a file.
    Args:
        model: Model architecture to load the weights into.
        model_filename: Filename of the saved model.
        device: Device to load the model onto.
    Returns:
        Model with loaded weights.
    """
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.to(device)
    return model

# Load the final model
model_filename = "final_model.pth"
model = load_model(model, model_filename, device)

# Set peak finding parameters
peak_height_threshold = 0.5

# Process samples
n_samples = 100000
#n_samples = len(dev_generator)

all_residual_p_arrival_times = []
all_residual_s_arrival_times = []

# Initialize counters for ground truth P and S labels
groundtruth_p_peaks = 0
groundtruth_s_peaks = 0

# Initialize counters for residuals smaller than 0.6 (absolute value)
count_residuals_p_under_0_6 = 0
count_residuals_s_under_0_6 = 0

for i in range(n_samples):
    sample = dev_generator[np.random.randint(len(dev_generator))]

    # Find peaks in the ground truth labels
    y_p_peaks, _ = find_peaks(sample["y"][0], height=peak_height_threshold, distance=100)
    y_s_peaks, _ = find_peaks(sample["y"][1], height=peak_height_threshold, distance=100)

    # Update the counters
    groundtruth_p_peaks += len(y_p_peaks)
    groundtruth_s_peaks += len(y_s_peaks)

    # Convert ground truth peak indices to time values
    sampling_rate = 100  # Samples per second (100 Hz)
    y_p_arrival_times = y_p_peaks / sampling_rate
    y_s_arrival_times = y_s_peaks / sampling_rate

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        pred = model(torch.tensor(sample["X"], device=device).unsqueeze(0))  # Add a fake batch dimension
        pred = pred[0].cpu().numpy()

    # Extract the probability distributions for P and S phases
    p_prob = pred[0]
    s_prob = pred[1]

    # Identify peaks in the probability distributions
    p_peaks, _ = find_peaks(p_prob, height=peak_height_threshold, distance=100)
    s_peaks, _ = find_peaks(s_prob, height=peak_height_threshold, distance=100)

    # Convert peak indices to time values
    p_arrival_times = p_peaks / sampling_rate
    s_arrival_times = s_peaks / sampling_rate

    # Calculate residuals for P and S peaks, keeping only the smallest one by absolute value
    for y_p_time in y_p_arrival_times:
        residual_p_arrival_times = p_arrival_times - y_p_time
        if len(residual_p_arrival_times) > 0:
            min_residual_p = residual_p_arrival_times[np.argmin(np.abs(residual_p_arrival_times))]
            all_residual_p_arrival_times.append(min_residual_p)
            if np.abs(min_residual_p) < 0.6:
                count_residuals_p_under_0_6 += 1
        
    for y_s_time in y_s_arrival_times:
        residual_s_arrival_times = s_arrival_times - y_s_time
        if len(residual_s_arrival_times) > 0:
            min_residual_s = residual_s_arrival_times[np.argmin(np.abs(residual_s_arrival_times))]
            all_residual_s_arrival_times.append(min_residual_s)
            if np.abs(min_residual_s) < 0.6:
                count_residuals_s_under_0_6 += 1

# Display the total counts of ground truth P and S peaks
print(f"Total ground truth P peaks: {groundtruth_p_peaks}")
print(f"Total ground truth S peaks: {groundtruth_s_peaks}")

# Display the counts of residuals under 0.6 seconds
print(f"Total P-phase residuals under 0.6s: {count_residuals_p_under_0_6}")
print(f"Total S-phase residuals under 0.6s: {count_residuals_s_under_0_6}")

# Plot the histogram of residual P peak arrival times
plt.figure(figsize=(10, 5))
counts_p, bins_p, patches_p = plt.hist(all_residual_p_arrival_times, bins=30, color='skyblue', edgecolor='black', range=(-1, 1))

# Add labels for the number counts on each column
for count, bin_, patch in zip(counts_p, bins_p, patches_p):
    plt.text(bin_ + (bins_p[1] - bins_p[0]) / 2, count, f'{int(count)}', ha='center', va='bottom')

# Print total pick count, mean, and standard deviation
total_picks_p = len(all_residual_p_arrival_times)
mean_p = np.mean(all_residual_p_arrival_times)
std_p = np.std(all_residual_p_arrival_times)
plt.text(0.95, 0.95, f'Total Picks: {total_picks_p}', ha='right', va='top', transform=plt.gca().transAxes)
print(f"P-phase Residuals: Mean = {mean_p:.4f}, Std = {std_p:.4f}")
print(f"Total detected P picks: {total_picks_p}")

plt.title('Histogram of Residual P Peak Arrival Times')
plt.xlabel('Residual P Arrival Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig("residual_p_histogram.png")  # Save the figure as a PNG file
plt.show()

# Plot the histogram of residual S peak arrival times
plt.figure(figsize=(10, 5))
counts_s, bins_s, patches_s = plt.hist(all_residual_s_arrival_times, bins=30, color='salmon', edgecolor='grey', range=(-1, 1))

# Add labels for the number counts on each column
for count, bin_, patch in zip(counts_s, bins_s, patches_s):
    plt.text(bin_ + (bins_s[1] - bins_s[0]) / 2, count, f'{int(count)}', ha='center', va='bottom')

# Print total pick count, mean, and standard deviation
total_picks_s = len(all_residual_s_arrival_times)
mean_s = np.mean(all_residual_s_arrival_times)
std_s = np.std(all_residual_s_arrival_times)
plt.text(0.95, 0.95, f'Total Picks: {total_picks_s}', ha='right', va='top', transform=plt.gca().transAxes)
print(f"S-phase Residuals: Mean = {mean_s:.4f}, Std = {std_s:.4f}")
print(f"Total detected S picks: {total_picks_s}")

plt.title('Histogram of Residual S Peak Arrival Times')
plt.xlabel('Residual S Arrival Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig("residual_s_histogram.png")  # Save the figure as a PNG file
plt.show()






