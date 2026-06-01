################################################################################
# Evaluate trained Fourier ViT runs and save the best reconstruction outputs.
# - Jialun Liu, LCN, UCL, 08.2025, jialun.liu.17@ucl.ac.uk
#
# Input:
#   Trained checkpoints, diffraction magnitude TIFF, support TIFF, optional phase TIFF.
#
# Output:
#   Best predicted amplitude/phase TIFFs, best prediction panel, chi-square histogram,
#   and radial diffraction profile.
################################################################################

import os
import numpy as np
import torch
import tifffile as tiff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
from fourier_vit_model import *
from train_single_input import *

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Configuration
# =============================================================================
# Experiment settings
NUM_RUNS = int(os.getenv("NUM_RUNS", "1"))
final_epoch = int(os.getenv("FINAL_EPOCH", "999"))
# ---------------------- Robust Path Setup ----------------------

# Step 1: Try to get env variables (set during multi-run mode)
dp_path_env = os.getenv("DP_PATH")
amp_path_env = os.getenv("AMP_PATH")
phase_path_env = os.getenv("PHASE_PATH")
output_dir_env = os.getenv("OUTPUT_DIR")

# Step 2: Fallback to manual path for standalone runs
default_base = "Jialun_test_data/LCMO-500"
dp_path = dp_path_env if dp_path_env else os.path.join(default_base, "dp_amp.tif")
amp_path = amp_path_env if amp_path_env else os.path.join(default_base, "support_original.tif")
phase_path = phase_path_env if phase_path_env else os.path.join(default_base, "phase.tif")
output_dir = output_dir_env if output_dir_env else "outputs/default"

# Step 3: Always infer base_dir dynamically from dp_path
base_dir = os.path.dirname(dp_path)

# Replace hardcoded 'checkpoints', 'results', etc., with output_dir-based paths:
checkpoint_dir = os.path.join(output_dir, "checkpoints")
results_dir = os.path.join(output_dir, "results")
hist_dir = os.path.join(output_dir, "histograms")

# Make them
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(hist_dir, exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    "img_size": 64,
    "patch_size": 4,
    "in_channels": 1,
    "embed_dim": 128,
    "num_layers": 10,
    "mlp_dim": 256,
    "dropout": 0.0,
    "use_multiscale": True,
}

# ---------------------- Plot display settings ----------------------
IMSHOW_KW = dict(origin='upper', interpolation='none', extent=[0, 64, 0, 64])

# =============================================================================
# Utility functions
# =============================================================================
def deramp_vit(phi_cvit, phi_iter, mask):
    cy = phi_cvit.shape[0] // 2
    cx = phi_cvit.shape[1] // 2 - 8
    offset = phi_cvit[cy, cx]
    p2 = phi_cvit.copy()
    p2[mask] = (p2[mask] - offset + np.pi) % (2 * np.pi) - np.pi
    p2[~mask] = np.nan
    return p2

# Metric functions
def calculate_chi_square(predicted_dp, target_dp):
    A = torch.abs(target_dp)
    B = torch.abs(predicted_dp)

    # RMS normalization
    A = A / torch.sqrt(torch.mean(A ** 2))
    B = B / torch.sqrt(torch.mean(B ** 2))

    numerator = torch.sum((A - B) ** 2)
    denominator = torch.sum(A ** 2)

    return (numerator / denominator).item()


def calculate_pcc(predicted_dp, target_dp):
    x = predicted_dp - predicted_dp.mean(dim=(-1, -2), keepdim=True)
    y = target_dp - target_dp.mean(dim=(-1, -2), keepdim=True)
    numerator = (x * y).sum(dim=(-1, -2))
    denominator = torch.sqrt((x ** 2).sum(dim=(-1, -2)) * (y ** 2).sum(dim=(-1, -2)))
    return (numerator / denominator).item()


def normalize_amp_inside_mask(arr, mask):
    arr = arr.copy()
    vals = arr[mask]
    if vals.size > 0:
        vmin = vals.min()
        vmax = vals.max()
    else:
        vmin, vmax = 0.0, 1.0
    norm = (arr - vmin) / (vmax - vmin + 1e-8)
    norm[~mask] = np.nan
    return norm


def radial_profile(img, center=None):
    y, x = np.indices(img.shape)
    if center is None:
        center = (img.shape[0] // 2, img.shape[1] // 2)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), img.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / (nr + 1e-8)
    return radial

# =============================================================================
# Model helpers
# =============================================================================
def build_model(params: dict) -> VisionTransformer:
    model = VisionTransformer(**params).to(device)
    model.eval()
    return model


def load_checkpoint(model: VisionTransformer, run_id: int) -> None:
    path = os.path.join(checkpoint_dir, f"single_dp_run_{run_id}.pth")
    model.load_state_dict(torch.load(path, map_location=device))

# =============================================================================
# Plotting helpers
# =============================================================================
def save_figure(fig, name):
    safe_name = os.path.normpath(name)
    fig.savefig(f"{safe_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_metric_histogram(data, best_idx, xlabel, title, out_path, color):
    """
    Histogram in blue, no grid, no best-run line.
    X-axis is chi² in [0, 3%].
    """
    from matplotlib.ticker import PercentFormatter, MultipleLocator

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot raw chi-square (0–0.03) and show as 0–3%
    ax.hist(data, bins=7, color="#DD8452", edgecolor="black", alpha=0.9)
    # green "#55A868"
    # orange "#DD8452"
    # purple "#6868B8"
    # light blue "#90D5FF"

    #ax.set_xlabel(xlabel, fontsize=13)
    #ax.set_ylabel("Frequency", fontsize=13)
    #ax.set_title(title, fontsize=14)

    # 0–3% range
    ax.set_xlim(0.0, 0.01)
    ax.xaxis.set_major_locator(MultipleLocator(0.002))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))  # 0.01 -> 1%
    ax.tick_params(labelsize=25, width=2.0, length=6)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Thicker box
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # no legend, no grid, no best-run marker
    save_figure(fig, out_path)

# ---------------------- 🔹 MAIN EVALUATION 🔹 ----------------------
if __name__ == "__main__":
    train_loader, _ = get_single_tif_loader(dp_path, amp_path, phase_path)
    dp_input, iter_phase, fixed_support = next(iter(train_loader))
    true_dp_input = dp_input.clone()  # same diffraction pattern used as target

    if dp_input.ndim == 3:
        dp_input = dp_input.unsqueeze(0)
    if true_dp_input.ndim == 3:
        true_dp_input = true_dp_input.unsqueeze(0)
    if fixed_support.ndim == 3:
        fixed_support = fixed_support.unsqueeze(0)
    if fixed_support.ndim == 2:
        fixed_support = fixed_support.unsqueeze(0).unsqueeze(0)

    true_dp_input = true_dp_input.to(device)
    dp_input = dp_input.to(device)
    fixed_support = fixed_support.to(device)
    prior_amp = make_prior(fixed_support).to(device)

    chi_square_list, pcc_list = [], []
    model = build_model(MODEL_PARAMS)

    # 🔹 Multi-run evaluation
    for run in range(NUM_RUNS):
        print(f"Evaluating Run {run}...")
        load_checkpoint(model, run)
        with torch.no_grad():
            #pred_amp, pred_phase = model(dp_input, fixed_support)
            raw_amp, pred_phase = model(dp_input, fixed_support, prior_amp=prior_amp)
            alpha = get_alpha(final_epoch)  # e.g. 999
            pred_amp = alpha * prior_amp + (1 - alpha) * raw_amp
            pred_dp_input = compute_fft_dp(pred_phase, pred_amp)

            chi = calculate_chi_square(pred_dp_input, true_dp_input)
            pcc = calculate_pcc(pred_dp_input, true_dp_input)

            chi_square_list.append(chi)
            pcc_list.append(pcc)

    chi_square_list = np.array(chi_square_list)
    pcc_list = np.array(pcc_list)

    best_run_idx = np.argmin(chi_square_list)
    worst_run_idx = np.argmax(chi_square_list)

    print(f"\nBest Run: {best_run_idx} (Chi-square={chi_square_list[best_run_idx]:.4f}, PCC={pcc_list[best_run_idx]:.4f})")
    print(f"Worst Run: {worst_run_idx} (Chi-square={chi_square_list[worst_run_idx]:.4f}, PCC={pcc_list[worst_run_idx]:.4f})")

    # Determine true histogram range
    chi_min = chi_square_list.min()
    chi_max = chi_square_list.max()
    print(f"Chi-square range: {chi_min:.6f} to {chi_max:.6f}")

    # --- Save Comparison Metrics to TXT with min/max for histogram ---
    summary_path = os.path.join(results_dir, "comparison.txt")
    with open(summary_path, "w") as f:
        f.write("# Phase Retrieval Quality Summary\n\n")
        f.write(f"Fourier_ViT_{best_run_idx} vs GT_DP:\n")
        f.write(f"    Chi-square: {chi_square_list[best_run_idx]:.6f}\n")
        f.write(f"    PCC:        {pcc_list[best_run_idx]:.6f}\n\n")
        f.write(f"# Histogram range for chi-square across runs\n")
        f.write(f"# min: {chi_min:.6f}\n")
        f.write(f"# max: {chi_max:.6f}\n")

    # --- Best Chi-square histogram ---
    plot_metric_histogram(
        chi_square_list,
        best_idx=best_run_idx,  # not used inside, but keeps API
        xlabel="Chi-square",
        title="",
        out_path=os.path.join(hist_dir, "chi_square_histogram"),
        color="#DD8452",
    )

    # --- Predict again using Best Run and Plot Comparison ---
    print(f"Generating final prediction figure for Best Run {best_run_idx}...")
    model = build_model(MODEL_PARAMS)
    load_checkpoint(model, best_run_idx)

    with torch.no_grad():
        raw_amp, pred_phase = model(dp_input, fixed_support, prior_amp=prior_amp)
        alpha = get_alpha(final_epoch)  # e.g. 999
        pred_amp = alpha * prior_amp + (1 - alpha) * raw_amp
        #pred_amp, pred_phase = model(dp_input, fixed_support)
        pred_dp_mag = compute_fft_dp(pred_phase, pred_amp)

    # Move tensors to CPU and numpy
    support_np = fixed_support.squeeze().cpu().numpy()
    mask = support_np > 0
    pred_amp_np = pred_amp.squeeze().cpu().numpy()
    pred_phase_np = pred_phase.squeeze().cpu().numpy()
    gt_dp_np = true_dp_input.squeeze().cpu().numpy()
    pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()
    del pred_amp, pred_phase, pred_dp_mag
    torch.cuda.empty_cache()

    # --- Amplitude normalization INSIDE the support, hide outside as NaN ---
    pred_amp_masked = normalize_amp_inside_mask(pred_amp_np, mask)

    # --- Phase offset using support, hide outside as NaN ---
    pred_phase_offset = deramp_vit(pred_phase_np, pred_phase_np, mask)

    # --- Normalize DP magnitude to [0,1] ---
    dp_pred_norm = pred_dp_np / (pred_dp_np.max() + 1e-8)
    dp_gt_norm = gt_dp_np / (gt_dp_np.max() + 1e-8)

    # --- Save best amplitude and phase as TIFFs ---
    tiff.imwrite(os.path.join(results_dir, "best_pred_amp.tif"), pred_amp_masked.astype(np.float32))
    tiff.imwrite(os.path.join(results_dir, "best_pred_phase.tif"), pred_phase_offset.astype(np.float32))

    # ----------------- FIGURE: Best predicted amp, phase, DP -----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ticks = [0, 16, 32, 48, 64]

    cmap_amp = colormaps["viridis"].copy()
    cmap_phase = colormaps["twilight_shifted"].copy()
    cmap_amp.set_bad("white")
    cmap_phase.set_bad("white")

    im0 = axes[0].imshow(pred_amp_masked, cmap=cmap_amp, vmin=0, vmax=1, **IMSHOW_KW)
    axes[0].set_title("Predicted Amplitude", fontsize=20)

    im1 = axes[1].imshow(pred_phase_offset, cmap=cmap_phase, vmin=-np.pi, vmax=np.pi, **IMSHOW_KW)
    axes[1].set_title("Predicted Phase", fontsize=20)

    im2 = axes[2].imshow(dp_pred_norm, cmap='turbo', vmin=0, vmax=1, **IMSHOW_KW)
    axes[2].set_title("Predicted DP Magnitude", fontsize=20)

    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=15)

    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.01)
    cbar0.set_label("Amplitude", fontsize=15)
    cbar0.ax.tick_params(labelsize=15)

    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.01)
    cbar1.set_label("Phase (rad)", fontsize=15)
    cbar1.set_ticks([-np.pi, 0, np.pi])
    cbar1.set_ticklabels(["-π", "0", "π"])
    cbar1.ax.tick_params(labelsize=15)

    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.01)
    cbar2.set_label("Intensity (a.u.)", fontsize=15)
    cbar2.ax.tick_params(labelsize=15)

    save_figure(fig, os.path.join(results_dir, f"best_prediction_run_{best_run_idx}"))

    # --- Save radial profile alone ---
    r_gt = radial_profile(dp_gt_norm)
    r_pred = radial_profile(dp_pred_norm)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_gt, label="GT", lw=3, color="#222222")
    ax.plot(r_pred, label="Fourier ViT", lw=3, color="#55A868")
    ax.set_yscale('log')

    ax.set_xlabel("Radial Distance (px)", fontsize=25)
    ax.set_ylabel("Diffraction (log)", fontsize=25)
    #ax.set_title("Radial Intensity Profile of DP", fontsize=20)

    # Optional grid – keep if you like it
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.legend(fontsize=18)

    # Thicker ticks and box (same style as histogram)
    ax.tick_params(labelsize=20, width=2.0, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    save_figure(fig, os.path.join(results_dir, "radial_profile_dp"))

    print(f"Best prediction figure saved under: {results_dir}")
    print(f"Best amplitude and phase TIFFs saved under: {results_dir}")
