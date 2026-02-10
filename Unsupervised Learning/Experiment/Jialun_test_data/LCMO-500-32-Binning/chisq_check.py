import numpy as np
import torch
import tifffile as tiff

# --- Load data ---
amp = tiff.imread("amp.tif").astype(np.float32)
phase = tiff.imread("phase.tif").astype(np.float32)
support = tiff.imread("support_original.tif").astype(np.float32)
dp_mag = tiff.imread("dp_amp.tif").astype(np.float32)
# --- Load DP as counts, convert to magnitude, max-normalize ---
"""
dp_counts = tiff.imread("LCMO-500_frame32_64x64.tif").astype(np.float32)
dp_counts = np.clip(dp_counts, 0, None)          # guard against any negatives
dp_mag = np.sqrt(dp_counts)                       # magnitude = sqrt(intensity)
"""
# --- Normalize DP ---
dp_mag /= (np.max(dp_mag) + 1e-8)

# --- Phase scaling ---
amp/= (np.max(amp) + 1e-8)
phase = (phase / np.max(phase)) * (2 * np.pi) - np.pi

# --- Apply support ---
amp_masked = amp * support
phase_masked = phase * support

# --- Density + FFT ---
density = amp_masked * np.exp(1j * phase_masked)
dp_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(density)))
dp_pred_mag = np.abs(dp_fft)
dp_pred_mag /= (np.max(dp_pred_mag) + 1e-8)

# --- Convert to torch ---
dp_pred = torch.tensor(dp_pred_mag[None, ...], dtype=torch.float32)
dp_true = torch.tensor(dp_mag[None, ...], dtype=torch.float32)

# --- Metric functions ---
def calculate_chi_square(predicted_dp, target_dp):
    predicted_norm = predicted_dp / (predicted_dp.amax(dim=(-1, -2), keepdim=True))
    target_norm = target_dp / (target_dp.amax(dim=(-1, -2), keepdim=True))
    chi = torch.sum((predicted_norm - target_norm) ** 2, dim=(-1, -2)) / (
        torch.sqrt(torch.sum(predicted_norm ** 2, dim=(-1, -2)) *
                   torch.sum(target_norm ** 2, dim=(-1, -2)))
    )
    return chi.item()

def calculate_pcc(predicted_dp, target_dp):
    x = predicted_dp - predicted_dp.mean(dim=(-1, -2), keepdim=True)
    y = target_dp - target_dp.mean(dim=(-1, -2), keepdim=True)
    num = (x * y).sum(dim=(-1, -2))
    denom = torch.sqrt((x ** 2).sum(dim=(-1, -2)) * (y ** 2).sum(dim=(-1, -2)))
    return (num / denom).item()

def calculate_chi1(predicted_dp, target_dp):
    A = torch.abs(target_dp)
    B = torch.abs(predicted_dp)

    A = A / A.max()
    B = B / B.max()

    numerator = torch.sum((A - B) ** 2)
    denominator = torch.sum(A ** 2)

    return (numerator / denominator).item()

def calculate_chi2(predicted_dp, target_dp):
    A = torch.abs(target_dp)
    B = torch.abs(predicted_dp)

    # RMS normalization
    A = A / torch.sqrt(torch.mean(A ** 2))
    B = B / torch.sqrt(torch.mean(B ** 2))

    numerator = torch.sum((A - B) ** 2)
    denominator = torch.sum(A ** 2)

    return (numerator / denominator).item()

# --- Calculate ---
chi_val = calculate_chi_square(dp_pred, dp_true)
chisq1 = calculate_chi1(dp_pred, dp_true)
chisq2 = calculate_chi2(dp_pred, dp_true)
pcc_val = calculate_pcc(dp_pred, dp_true)

print(f"Chi-square: {chi_val:.6f}")
print(f"Chi-square1: {chisq1:.6f}")
print(f"Chi-square2: {chisq2:.6f}")
print(f"PCC:        {pcc_val:.6f}")
