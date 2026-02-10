import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter, zoom
from skimage import exposure
import matplotlib.pyplot as plt
import os

def save_tiff(data, filename, normalize_range=False):
    if normalize_range:
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 65535
    data = np.clip(data, 0, 65535).astype(np.uint16)
    tiff.imwrite(filename, data)

def create_softer_amplitude(amplitude, sigma=1.0):
    """
    Generate amplitude with:
    - Flat core (1)
    - Soft Gaussian falloff at boundary
    - Hard shape clamping
    - Final normalization
    """
    support_mask = (amplitude > 0).astype(np.float32)

    # Step 1: Apply stronger Gaussian smoothing
    soft_amp = gaussian_filter(support_mask, sigma=sigma)
    soft_amp = soft_amp ** 3

    # Step 2: Clamp with original shape (prevent leakage)
    soft_amp *= support_mask

    # Step 3: Normalize
    soft_amp = exposure.rescale_intensity(soft_amp, out_range=(0, 1))

    return soft_amp


def intensity_to_amplitude(intensity_tif_path, save_path=None):

    intensity = tiff.imread(intensity_tif_path).astype(np.float32)
    # safety clip
    intensity = np.clip(intensity, 0, None)

    amplitude = np.sqrt(intensity)
    amp_norm = amplitude / (np.max(amplitude) + 1e-8)
    amp_uint16 = (amp_norm * 65535).astype(np.uint16)

    if save_path is None:
        save_path = "dp_amp.tif"

    tiff.imwrite(save_path, amp_uint16)
    print(f"Saved amplitude TIFF to {save_path}")

    return amp_uint16


def resize_support(support, scale=1.5):
    """
    Resize binary support mask using zoom. Scale >1 makes it larger.
    The result will be padded or cropped to match original shape.
    """
    original_shape = support.shape
    zoomed = zoom(support.astype(np.float32), scale, order=1)  # 🔧 fix type issue

    zoomed_shape = zoomed.shape
    output = np.zeros_like(support, dtype=np.float32)

    # Calculate crop or pad indices
    y_offset = (zoomed_shape[0] - original_shape[0]) // 2
    x_offset = (zoomed_shape[1] - original_shape[1]) // 2

    if scale >= 1.0:
        output[...] = zoomed[y_offset:y_offset + original_shape[0],
                             x_offset:x_offset + original_shape[1]]
    else:
        output[y_offset:y_offset + zoomed_shape[0],
               x_offset:x_offset + zoomed_shape[1]] = zoomed

    return (output > 0).astype(np.float32)  # return binary mask


if __name__ == "__main__":

    # Process soft support
    amp_in = tiff.imread("support_original.tif").astype(np.float32)
    amp_out = create_softer_amplitude(amp_in, sigma=1.5)
    #save_tiff(amp_out, "Sup_A_phased-support.tif", normalize_range=True)

    support_scale = 1.5
    support_mask = (amp_in > 0).astype(np.float32)
    support_large = resize_support(support_mask, scale=support_scale)
    amp_large_soft = create_softer_amplitude(support_large, sigma=1.5)
    #save_tiff(amp_large_soft, "support_large.tif", normalize_range=True)

    # Convert intensity to amplitude
    amplitude_from_intensity = intensity_to_amplitude("STO_1039082-slice_crop.tif")

    # Visualize both
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(amp_in, cmap='viridis')
    axes[0].set_title("Original Amplitude")
    axes[0].axis('off')

    axes[1].imshow(amp_out, cmap='viridis')
    axes[1].set_title("Improved Soft Amplitude")
    axes[1].axis('off')

    axes[2].imshow(amplitude_from_intensity, cmap='viridis')
    axes[2].set_title("Amplitude from Intensity")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


