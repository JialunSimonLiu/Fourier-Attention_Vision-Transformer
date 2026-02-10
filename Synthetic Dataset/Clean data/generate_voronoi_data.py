
################################################################################
# multi_domains_simulation.py
# Simulations of multi-domains crystals with Voronoi partitions,
# and corresponding diffraction patterns
# - Jialun Liu, LCN, UCL, 11.01.2024, jialun.liu.17@ucl.ac.uk
# output movie of series of domains for UMinn test  IKR 9/2024
# option to switch to list of input (x,y,ph) values instead of random
# Defined the assign_strong_phase_values function LJL 14.11.2024
# ensured strong phase for adjacent seeds by the cut value
################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import tifffile as tiff
import h5py
from scipy.fft import fft2, fftshift
from skimage.feature import peak_local_max
from skimage import img_as_float
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter
from skimage import exposure
import random
np.random.seed(1)
random.seed(1)

#################### Parameters ####################
N = 1 # Number of crystals
num_seeds = 10 # Number of domains
grid_size = 32  # Size of the crystal
pad_size = 64  # Size of the sample
zoom_size = 32  # Size of the zoomed diffraction pattern
clash = 3  # minimum overlap distance of seeds (originally 3)
cut = 0.5  # radians minimum separation of allowed phase values
N_maxima = 10 # Fixed number of DP maxima for the batch training
prominence = 0.001  # Adjust the prominence value as needed between 0 and 1
threshold = 0.01 # Remove the pixels below the threshold amplitude

# ---------------------- 🔹 Function to Save 16-bit TIFF ----------------------
def save_tiff(data, filename, normalize_range=False, phase=False):
    """
    Saves data as a 16-bit TIFF image.
    Normalize data to [0, 65535] for proper 16-bit storage.
    Rescale phase from [-π, π] to [0, 65535].
    """
    if phase:
        data = (data + np.pi) * (65535 / (2 * np.pi))  # Convert [-π, π] to [0, 65535]
    elif normalize_range:
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 65535  # Normalize to [0, 65535]

    data = np.clip(data, 0, 65535).astype(np.uint16)  # Ensure correct dtype
    tiff.imwrite(filename, data)


#################### Functions ####################

def generate_circular_seeds(num_seeds, radius, seed):
    np.random.seed(seed)
    seeds = []
    while len(seeds) < num_seeds:
        x, y = np.random.rand(2) * 2 - 1  # Generate points
        if x ** 2 + y ** 2 <= 1:  # Check if the point is inside the circle
            new_seed = np.array([x * radius + radius, y * radius + radius])
            # Check for overlapping with existing seeds
            overlap = False
            for seed in seeds:
                if np.linalg.norm(new_seed - seed[:2]) < clash:
                    overlap = True
                    break
            if not overlap:
                phase = np.random.uniform(-np.pi, np.pi)
                seeds.append([new_seed[0], new_seed[1], phase])
    return np.array(seeds)

def discrete_voronoi(seeds, grid_size):
    #Generates a discrete Voronoi diagram on a grid and pads it to a specified size.

    # Creating a meshgrid for the coordinates
    x = np.arange(0, grid_size, 1)
    y = np.arange(0, grid_size, 1)
    xx, yy = np.meshgrid(x, y)
    # Calculate the distance from each point in the grid to each seed
    distances = np.sqrt((xx[..., np.newaxis] - seeds[:, 0])**2 + (yy[..., np.newaxis] - seeds[:, 1])**2)
    # Find the nearest seed for each point in the grid
    nearest_seed_index = np.argmin(distances, axis=2)
    return nearest_seed_index

def create_circular_voronoi_diagram(seeds, grid_size, pad_size):
    # Creates and pads a circular Voronoi diagram.

    voronoi_diagram = discrete_voronoi(seeds, grid_size)

    # Apply circular mask
    radius = grid_size // 2
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2
    voronoi_diagram[~mask] = 0 # Mark outside of the circle

    # Pad the diagram
    padded_voronoi = pad_voronoi_diagram(voronoi_diagram, grid_size, pad_size)
    return padded_voronoi

def pad_voronoi_diagram(nearest_seed_index, grid_size, pad_size):
    # Pad the result to the desired size
    pad_width = (pad_size - grid_size) // 2
    padded_voronoi = np.pad(nearest_seed_index, pad_width, mode='constant', constant_values=0)
    return padded_voronoi


def assign_strong_phase_values(nearest_seed_index, num_seeds, grid_size, pad_size, seeds, cut=cut):
    min_phase, max_phase = -np.pi, np.pi
    seed_positions = np.array([[s[0], s[1]] for s in seeds])

    # Compute Voronoi neighbors
    vor = Voronoi(seed_positions)
    neighbors = [[] for _ in range(num_seeds)]

    # Determine neighbors using Voronoi ridge points
    for ridge in vor.ridge_points:
        p1, p2 = ridge
        neighbors[p1].append(p2)
        neighbors[p2].append(p1)

    # Helper function for circular phase difference
    def circular_diff(phase1, phase2):
        diff = np.abs(phase1 - phase2)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        return diff

    # Assign phases with strict separation enforcement
    assigned_phases = np.full(num_seeds, None, dtype=float)  # Initialise phases for all seeds as None
    # Identify the central seed (closest to the center of the circular domain)
    circle_center = np.array([grid_size / 2, grid_size / 2])
    distances_to_center = np.linalg.norm(seed_positions - circle_center, axis=1)
    central_seed_index = np.argmin(distances_to_center)

    # Force central seed to have phase = 0.0
    assigned_phases[central_seed_index] = 0.0

    for seed_index in range(num_seeds):
        if seed_index == central_seed_index:
            continue  # Skip phase assignment for the central seed

        for attempt in range(1000):
            candidate_phase = np.random.uniform(min_phase, max_phase)
            # Check if this phase is valid by comparing it with neighbours' phases
            valid_phase = True  # Assume the phase is valid initially
            for neighbor in neighbors[seed_index]:
                if assigned_phases[neighbor] is not None:
                    # Calculate the circular difference between the candidate and neighbor's phase
                    phase_difference = circular_diff(candidate_phase, assigned_phases[neighbor])
                    # If the difference is less than the required 'cut', the phase is invalid
                    if phase_difference < cut:
                        valid_phase = False
                        break
            if valid_phase:  # If the candidate phase is valid, assign it to the current seed
                assigned_phases[seed_index] = candidate_phase
                break
        # If no valid phase was found after 1000 attempts, raise an error
        if assigned_phases[seed_index] is None:
            raise ValueError(
                f"Failed to assign a valid phase for seed {seed_index} with required separation of {cut}"
            )

    # Map the assigned phases to the grid based on nearest seed index from discrete Voronoi
    phase_assigned = assigned_phases[nearest_seed_index]
    labels = nearest_seed_index

    # Apply circular masking to ensure the structure remains a circular crystal
    radius = grid_size // 2
    circle_center = pad_size / 2
    for i in range(pad_size):
        for j in range(pad_size):
            if (i - circle_center) ** 2 + (j - circle_center) ** 2 > radius ** 2:
                phase_assigned[i, j] = 0

    # Debugging outputs
    #print(f"Assigned phases: {assigned_phases}")
    return phase_assigned, labels, assigned_phases

def create_amplitude_function(grid_size, pad_size):
    """
    Creates a 2D projection of a sphere to use as an amplitude function.
    """
    # Calculate parameters
    radius = grid_size / 2
    pad_center = pad_size / 2

    # Create a meshgrid for the padded size
    y, x = np.ogrid[:pad_size, :pad_size]
    distance_from_center = np.sqrt((x - pad_center) ** 2 + (y - pad_center) ** 2)

    # Create the Gaussian decay only for distances within the grid radius
    amplitude_grid = np.zeros((pad_size, pad_size), dtype=np.float32)
    mask = distance_from_center <= radius
    amplitude_grid[mask] = np.exp(-(distance_from_center[mask] ** 2) / (2 * (radius / 2) ** 2))

    # Normalize the amplitude
    amplitude_grid /= np.max(amplitude_grid)

    return amplitude_grid

def create_density_function(amplitude, phase):
    """
    Combines the amplitude and phase to form a complex-valued function.
    """
    if np.iscomplexobj(phase):
        phase = np.real(phase)
    return amplitude * np.exp(phase * 1j )

def fourier_transform(image):
    """
    Computes the Fourier transform of an image and returns its magnitude (diffraction pattern).
    """
    # Compute the Fourier transform
    ft = np.fft.fft2(image)
    ft_shifted = np.fft.fftshift(ft) # Contains full information of the DP

    # Compute the magnitude (diffraction pattern)
    amplitude = np.abs(ft_shifted)

    return amplitude

def remove_zero_intensity(diffraction_pattern, threshold = threshold):
    return np.where(diffraction_pattern > threshold, diffraction_pattern, 0)

def find_maxima_with_prominence(image, prominence, min_distance=1, threshold_abs=0, grid_size=1):
    """Detect maxima in an image, ensuring the output contains exactly 10 peaks."""
    # Normalize the image to [0,1]
    image = img_as_float(image)

    # Find all local maxima
    coordinates = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold_abs)
    maxima = []

    for coord in coordinates:
        x, y = coord
        local_max = image[x, y]
        # Define the surrounding region based on grid size
        surrounding = image[max(0, x - grid_size):x + grid_size + 1, max(0, y - grid_size):y + grid_size + 1]

        # Check the prominence condition
        if local_max - np.max(surrounding[surrounding != local_max]) >= prominence:
            maxima.append([x, y, round(local_max, 4)])

    # Handle the case where no maxima are found
    if not maxima:
        maxima = np.zeros((0, 3))

    maxima = np.array(maxima)

    # Trim or pad to ensure exactly 10 peaks
    if maxima.shape[0] > N_maxima:
        # Trim excess peaks
        maxima = maxima[:N_maxima]
    elif maxima.shape[0] < N_maxima:
        # Pad with zeros if fewer peaks are found
        padding = np.zeros((N_maxima - maxima.shape[0], 3))  # Ensure consistent shape for padding
        maxima = np.vstack((maxima, padding))

    return maxima


def save_as_tiff(data, filename):
    tiff.imsave(f'{filename}.tif', data)

#################### Simulation and Output ####################
def save_data_vit(crystal_amp, crystal_pha, dp, group_idx):
    """
    Save crystal amplitude, phase, and DP as .h5 and .txt files.
    Appends multiple groups for efficient storage.
    """

    """# Ensure correct shape
    assert crystal_amp.shape == (32, 32), f"Expected (32,32), got {crystal_amp.shape}"
    assert crystal_pha.shape == (32, 32), f"Expected (32,32), got {crystal_pha.shape}"
    assert dp.shape == (32, 32), f"Expected (32,32), got {dp.shape}"
    """

    # File paths
    h5_files = {
        "crystal_amp": f"crystal_amp.h5",
        "crystal_pha": f"crystal_pha.h5",
        "dp": f"dp.h5"
    }

    """txt_files = {
        "crystal_amp": f"crystal_amp.txt",
        "crystal_pha": f"crystal_pha.txt",
        "dp": f"dp.txt"
    }"""

    # Helper function to save HDF5 and append text
    def save_h5_and_txt(data, name):
        # Append to HDF5 (group-based structure)
        with h5py.File(h5_files[name], 'a') as h5_file:
            h5_file.create_dataset(f"{name}_group_{group_idx}", data=data, compression="gzip")

        # Append to TXT
        #with open(txt_files[name], 'a') as txt_file:
            #np.savetxt(txt_file, data, fmt='%.6f', header=f"{name} Group {group_idx}", comments='')

    # Save all datasets
    """save_h5_and_txt(crystal_amp, "crystal_amp")
    save_h5_and_txt(crystal_pha, "crystal_pha")
    save_h5_and_txt(dp, "dp")"""

    # 🔹 Save GT data as 16-bit TIFFs
    save_tiff(crystal_amp, f"gt_amp_{group_idx}.tif", normalize_range=True)
    save_tiff(crystal_pha, f"gt_phase_{group_idx}.tif", phase=True)
    save_tiff(dp, f"gt_dp_mag_{group_idx}.tif", normalize_range=True)


    #print(f"Saved group {group_idx} to {', '.join(h5_files.values())}")


# === Updated Seed Generator Respecting Support ===
def generate_masked_seeds(num_seeds, mask, seed=42):
    np.random.seed(seed)
    h, w = mask.shape
    valid_coords = np.argwhere(mask)
    seeds = []

    while len(seeds) < num_seeds:
        idx = np.random.choice(len(valid_coords))
        y, x = valid_coords[idx]
        new_seed = np.array([x, y])
        overlap = False
        for s in seeds:
            if np.linalg.norm(new_seed - s[:2]) < clash:
                overlap = True
                break
        if not overlap:
            phase = np.random.uniform(-np.pi, np.pi)
            seeds.append([new_seed[0], new_seed[1], phase])

    return np.array(seeds)

# === Crescent Cut Mask Function with Soft Edge ===
def create_crescent_cut_mask(grid_size, pad_size):
    radius = grid_size // 2
    center = pad_size // 2
    y, x = np.ogrid[:pad_size, :pad_size]

    # Main circular support
    base_mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2

    # Crescent cut parameters
    cut_radius = radius / 3
    cut_depth = max(1, int(radius / 10))
    cut_center_x = center + radius - cut_depth
    cut_mask = (x - cut_center_x) ** 2 + (y - center) ** 2 < cut_radius ** 2

    # Final support
    support_mask = base_mask & ~cut_mask
    return support_mask.astype(float)

# === Updated Soft Amplitude Function ===

def create_flat_core_amplitude(grid_size, pad_size, flat_ratio=0.8, sigma=1.0):
    """
    v2: Soft-edged amplitude mask with cosine ramp and crescent cut,
    followed by Gaussian smoothing for soft boundaries.
    """
    radius = grid_size / 2
    center = pad_size / 2
    y, x = np.ogrid[:pad_size, :pad_size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)

    # Flat core with cosine ramp to edge
    amp = np.zeros((pad_size, pad_size), dtype=np.float32)
    flat_core = radius * flat_ratio
    edge_width = radius * (1 - flat_ratio)

    amp[dist <= flat_core] = 1.0
    edge_mask = (dist > flat_core) & (dist <= radius)
    ramp = (dist[edge_mask] - flat_core) / edge_width
    amp[edge_mask] = 0.5 * (1 + np.cos(np.pi * ramp))

    # Crescent cut (binary)
    cut_radius = radius / 3
    cut_depth = max(1, int(radius / 10))
    cut_center_x = center + radius - cut_depth
    cut_dist = np.sqrt((x - cut_center_x)**2 + (y - center)**2)
    amp[cut_dist < cut_radius] = 0.0

    # Gaussian smoothing
    soft_amp = gaussian_filter(amp, sigma=sigma)
    soft_amp *= support_mask

    # Normalize to [0, 1]
    amp = exposure.rescale_intensity(amp, out_range=(0, 1))

    return soft_amp


# === Updated Seed Generator (Respects Mask) ===
def generate_masked_seeds(num_seeds, mask, seed=42):
    np.random.seed(seed)
    h, w = mask.shape
    valid_coords = np.argwhere(mask > 0)
    seeds = []

    while len(seeds) < num_seeds:
        idx = np.random.choice(len(valid_coords))
        y, x = valid_coords[idx]
        new_seed = np.array([x, y])
        if all(np.linalg.norm(new_seed - s[:2]) >= clash for s in seeds):
            phase = np.random.uniform(-np.pi, np.pi)
            seeds.append([new_seed[0], new_seed[1], phase])

    return np.array(seeds)

# === Updated Voronoi Generator (Respects Cut) ===
def create_crescent_voronoi_diagram(seeds, pad_size, mask):
    x = np.arange(0, pad_size, 1)
    y = np.arange(0, pad_size, 1)
    xx, yy = np.meshgrid(x, y)

    distances = np.full((pad_size, pad_size, len(seeds)), np.inf)
    for i, (sx, sy, _) in enumerate(seeds):
        distances[..., i] = np.sqrt((xx - sx)**2 + (yy - sy)**2)

    nearest_seed_index = np.argmin(distances, axis=2)
    nearest_seed_index[mask == 0] = 0  # Apply mask before phase assignment
    return nearest_seed_index

# === Main Simulation Loop ===
for j in range(0, N):
    support_mask = create_crescent_cut_mask(grid_size, pad_size)
    amplitude = create_flat_core_amplitude(grid_size, pad_size)
    seeds = generate_masked_seeds(num_seeds, support_mask, seed=j)
    padded_voronoi_diagram = create_crescent_voronoi_diagram(seeds, pad_size, support_mask)

    phase_assigned_voronoi, labels, assigned_phases = assign_strong_phase_values(
        padded_voronoi_diagram, num_seeds, grid_size, pad_size, seeds
    )
    for i, phase in enumerate(assigned_phases):
        seeds[i][2] = phase
    phase_assigned_voronoi = np.array(phase_assigned_voronoi, dtype=float)
    support_mask = create_crescent_cut_mask(grid_size, pad_size)
    phase_assigned_voronoi[~support_mask.astype(bool)] = 0

    density_function = create_density_function(amplitude, phase_assigned_voronoi)
    diffraction_pattern = fourier_transform(density_function)
    diffraction_pattern_norm = diffraction_pattern / np.max(diffraction_pattern)

    full_diffraction = diffraction_pattern_norm
    thresholded_diffraction = remove_zero_intensity(full_diffraction)
    maxima_diffraction = find_maxima_with_prominence(full_diffraction, prominence)

    save_data_vit(amplitude, phase_assigned_voronoi, full_diffraction, group_idx=j + 1)

    # Plot the amplitude of the crystal
    plt.figure()
    plt.imshow(amplitude, cmap='viridis')
    plt.colorbar()
    plt.title(f'Crystal Amplitude (Frame {j})')
    plt.show()

    # Plot the phase of the crystal
    plt.figure()
    plt.imshow(phase_assigned_voronoi, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.title(f'Crystal Phase (Frame {j})')
    plt.show()

    # Plot the full diffraction pattern
    plt.figure()
    plt.imshow(full_diffraction, cmap='jet')
    plt.colorbar()
    plt.title(f'Full Diffraction Pattern (Frame {j})')
    plt.show()