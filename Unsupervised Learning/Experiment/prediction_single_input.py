################################################################################
# Fourier ViT prediction file
# - Jialun Liu, LCN, UCL, 11-12.2025, jialun.liu.17@ucl.ac.uk
################################################################################
import os
import numpy as np
import torch
import tifffile as tiff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader, Dataset
from real_cnnvit_fourier_attention_amp_generalized import *
from train_single_input import *

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Configuration
# =============================================================================
# Experiment settings
NUM_RUNS = 100
final_epoch = 999
# ---------------------- Robust Path Setup ----------------------

# I: Try to get env variables (set during multi-run mode)
dp_path_env = os.getenv("DP_PATH")
amp_path_env = os.getenv("AMP_PATH")
phase_path_env = os.getenv("PHASE_PATH")
output_dir_env = os.getenv("OUTPUT_DIR")

# II: Fallback to manual path for standalone runs
default_base = "Jialun_test_data/LCMO-500"
dp_path = dp_path_env if dp_path_env else os.path.join(default_base, "dp_amp.tif")
amp_path = amp_path_env if amp_path_env else os.path.join(default_base, "support_original.tif")
phase_path = phase_path_env if phase_path_env else os.path.join(default_base, "phase.tif")
output_dir = output_dir_env if output_dir_env else "outputs/default"

# III: Always infer base_dir dynamically from dp_path
base_dir = os.path.dirname(dp_path)
iter_amp_path = os.path.join(base_dir, "amp.tif")
iter_amp = tiff.imread(iter_amp_path).astype(np.float32)
iter_amp = iter_amp / (np.max(iter_amp))
iter_amp = torch.tensor(iter_amp).unsqueeze(0).to(device)
# Load support_original
support_path = os.path.join(base_dir, "support_original.tif")
support_np = tiff.imread(support_path).astype(np.float32)
support_mask = torch.tensor((support_np > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0).to(device)

# Replace hardcoded 'checkpoints', 'results', etc., with output_dir-based paths:
checkpoint_dir = os.path.join(output_dir, "checkpoints")
results_dir = os.path.join(output_dir, "results")
loss_dir = os.path.join(output_dir, "loss_plots")
hist_dir = os.path.join(output_dir, "histograms")

# Make them
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
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

# =============================================================================
# Dataset and DataLoader
# =============================================================================
# =============================================================================
# Utility functions
# =============================================================================
# Apply phase offset correction
def apply_phase_offset(phase, mask):
    cy, cx = phase.shape[0] // 2, phase.shape[1] // 2
    offset = phase[cy, cx]
    p2 = phase.copy()
    p2[mask] = (p2[mask] - offset + np.pi) % (2 * np.pi) - np.pi
    p2[~mask] = np.nan
    return p2

# Metric functions
def calculate_chi_square(predicted_dp, target_dp):
    """
    predicted_norm = predicted_dp / (predicted_dp.amax(dim=(-1, -2), keepdim=True) )
    target_norm = target_dp / (target_dp.amax(dim=(-1, -2), keepdim=True))
    chi_square = torch.sum((predicted_norm - target_norm) ** 2, dim=(-1, -2)) / (
        torch.sqrt(torch.sum(predicted_norm ** 2, dim=(-1, -2)) * torch.sum(target_norm ** 2, dim=(-1, -2)))
    )
    return chi_square.item()
    """
    A = torch.abs(target_dp)
    B = torch.abs(predicted_dp)

    # RMS normalization
    A = A / torch.sqrt(torch.mean(A ** 2))
    B = B / torch.sqrt(torch.mean(B ** 2))

    numerator = torch.sum((A - B) ** 2)
    denominator = torch.sum(A ** 2)

    return (numerator / denominator).item()


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

def calculate_pcc(predicted_dp, target_dp):
    x = predicted_dp - predicted_dp.mean(dim=(-1, -2), keepdim=True)
    y = target_dp - target_dp.mean(dim=(-1, -2), keepdim=True)
    numerator = (x * y).sum(dim=(-1, -2))
    denominator = torch.sqrt((x ** 2).sum(dim=(-1, -2)) * (y ** 2).sum(dim=(-1, -2)))
    return (numerator / denominator).item()

def apply_mask_and_evaluate(amp, phase, support_mask, dp_gt):
    amp = amp.to(device)
    phase = phase.to(device)
    support_mask = support_mask.to(device)
    dp_gt = dp_gt.to(device)
    amp_masked = amp * support_mask
    phase_masked = phase * support_mask
    dp_pred = compute_fft_dp(phase_masked, amp_masked)
    chi = calculate_chi_square(dp_pred, dp_gt)
    pcc = calculate_pcc(dp_pred, dp_gt)
    return chi, pcc, dp_pred

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
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.close(fig)
    ax.hist(data, bins=50, color=color, edgecolor='black', alpha=0.8)
    ax.axvline(data[best_idx], color='red', linestyle='--',
               label=f"Best Run {best_idx}")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=11)
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
    if fixed_support.ndim == 3:
        fixed_support = fixed_support.unsqueeze(1)

    true_dp_input = true_dp_input.to(device)
    dp_input = dp_input.to(device)
    fixed_support = fixed_support.to(device)
    prior_amp = make_prior(fixed_support).to(device)

    chi_square_list, pcc_list = [], []
    model = build_model(MODEL_PARAMS)

    # Multi-run evaluation
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

    # ===================== Save Comparison Metrics to TXT ======================
    # Fix phase scaling for iterative result
    iter_phase_np = tiff.imread(os.path.join(base_dir, "phase.tif")).astype(np.float32)
    iter_phase = torch.tensor((iter_phase_np / 65535) * (2 * np.pi) - np.pi, dtype=torch.float32).to(device)
    chisq_trad_iter_vs_gt, pcc_trad_iter_vs_gt, iter_dp_tensor = apply_mask_and_evaluate(iter_amp, iter_phase,
                                                                                         support_mask, true_dp_input)

    chisq_pred_vs_gt = chi_square_list[best_run_idx]
    pcc_pred_vs_gt = pcc_list[best_run_idx]

    # recompute DP for the *best run* before chi1/chi2
    model = build_model(MODEL_PARAMS)
    load_checkpoint(model, best_run_idx)
    with torch.no_grad():
        raw_amp, best_pred_phase = model(dp_input, fixed_support, prior_amp=prior_amp)
        alpha = get_alpha(final_epoch)  # e.g. 999
        best_pred_amp = alpha * prior_amp + (1 - alpha) * raw_amp
        #best_pred_amp, best_pred_phase = model(dp_input, fixed_support)
        best_pred_dp_mag = compute_fft_dp(best_pred_phase, best_pred_amp)

    chi1 = calculate_chi1(best_pred_dp_mag, true_dp_input)
    chi2 = calculate_chi2(best_pred_dp_mag, true_dp_input)
    chi1_iter = calculate_chi1(iter_dp_tensor, true_dp_input)
    chi2_iter = calculate_chi2(iter_dp_tensor, true_dp_input)

    # Save as TXT
    summary_path = os.path.join(results_dir, "comparison.txt")
    with open(summary_path, "w") as f:
        f.write("# Phase Retrieval Quality Summary\n\n")
        f.write("Traditional_iterative vs GT_DP:\n")
        f.write(f"    Chi-square: {chisq_trad_iter_vs_gt:.6f}\n")
        f.write(f"    PCC:        {pcc_trad_iter_vs_gt:.6f}\n\n")
        f.write(f"    Chi1 (Max-Norm):     {chi1_iter:.6f}\n")
        f.write(f"    Chi2 (RMS-Norm):     {chi2_iter:.6f}\n")
        f.write(f"C-ViT_{best_run_idx} vs GT_DP:\n")
        f.write(f"    Chi-square: {chisq_pred_vs_gt:.6f}\n")
        f.write(f"    PCC:        {pcc_pred_vs_gt:.6f}\n")
        f.write(f"    Chi1 (Max-Norm):     {chi1:.6f}\n")
        f.write(f"    Chi2 (RMS-Norm):     {chi2:.6f}\n")

    # Upgraded: Chi-square histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.close(fig)
    ax.hist(chi_square_list, bins=50, color="#4C72B0", edgecolor='black', alpha=0.8)
    ax.axvline(chi_square_list[best_run_idx], color='red', linestyle='--', label=f"Best Run {best_run_idx}")
    ax.set_xlabel("Chi-square", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Distribution of Chi-square Across Runs", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=11)
    save_figure(fig, os.path.join(hist_dir, "chi_square_histogram"))

    # Upgraded: PCC histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.close(fig)
    ax.hist(pcc_list, bins=50, color="#55A868", edgecolor='black', alpha=0.8)
    ax.axvline(pcc_list[best_run_idx], color='red', linestyle='--', label=f"Best Run {best_run_idx}")
    ax.set_xlabel("Pearson Correlation Coefficient", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Distribution of PCC Across Runs", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=11)
    save_figure(fig, os.path.join(hist_dir, "pcc_histogram"))

    # Upgraded: Best loss curve
    loss_history = np.load(os.path.join(loss_dir, f"loss_per_epoch_run_{best_run_idx}.npy"))
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.close(fig)
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color="#C44E52")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss (log scale)", fontsize=13)
    ax.set_title(f"Loss Curve of Best Run ({best_run_idx})", fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=11)
    save_figure(fig, os.path.join(loss_dir, f"best_loss_curve_run_{best_run_idx}"))

    # Predict again using Best Run and Plot Comparison
    print(f"Generating final comparison figure for Best Run {best_run_idx}...")
    model = build_model(MODEL_PARAMS)
    load_checkpoint(model, best_run_idx)

    with torch.no_grad():
        raw_amp, pred_phase = model(dp_input, fixed_support, prior_amp=prior_amp)
        alpha = get_alpha(final_epoch)  # e.g. 999
        pred_amp = alpha * prior_amp + (1 - alpha) * raw_amp
        #pred_amp, pred_phase = model(dp_input, fixed_support)
        pred_dp_mag = compute_fft_dp(pred_phase, pred_amp)

    # Move tensors to CPU and numpy
    amp_np = fixed_support.squeeze().cpu().numpy()
    pred_amp_np = pred_amp.squeeze().cpu().numpy()
    gt_phase_np = iter_phase.squeeze().cpu().numpy()
    pred_phase_np = pred_phase.squeeze().cpu().numpy()
    gt_dp_np = true_dp_input.squeeze().cpu().numpy()
    pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()
    del pred_amp, pred_phase, pred_dp_mag
    torch.cuda.empty_cache()

    # Top 5 Best Runs by Chi-square
    print("Generating reproducibility check for Top 5 Chi-square Runs...")

    # Use ground truth amplitude mask for all phase offsetting
    mask = amp_np > 0

    def apply_phase_offset(phase, mask):
        cy, cx = phase.shape[0] // 2, phase.shape[1] // 2
        offset = phase[cy, cx]
        p2 = phase.copy()
        p2[mask] = (p2[mask] - offset + np.pi) % (2 * np.pi) - np.pi
        p2[~mask] = 0.0
        return p2

    def deramp(phase, mask):
        cy, cx = phase.shape[0] // 2, phase.shape[1] // 2
        offset = phase[cy, cx]
        p2 = phase.copy()
        p2[mask] = (p2[mask] - offset + np.pi) % (2 * np.pi) - np.pi
        p2[~mask] = np.nan
        return p2

    top5_indices = np.argsort(chi_square_list)[:5]
    fig, ax = plt.subplots(3, 5, figsize=(25, 15), constrained_layout=True)
    ticks = [0, 16, 32, 48, 64]

    model = build_model(MODEL_PARAMS)

    for col, run_idx in enumerate(top5_indices):
        print(f"Run {run_idx} (Chi-square = {chi_square_list[run_idx]:.5f})")
        load_checkpoint(model, run_idx)

        with torch.no_grad():
            raw_amp, pred_phase = model(dp_input, fixed_support, prior_amp=prior_amp)
            alpha = get_alpha(final_epoch)  # e.g. 999
            pred_amp = alpha * prior_amp + (1 - alpha) * raw_amp
            #pred_amp, pred_phase = model(dp_input, fixed_support)
            pred_dp_mag = compute_fft_dp(pred_phase, pred_amp)

        # Convert to numpy
        pred_amp_np = pred_amp.squeeze().cpu().numpy()
        pred_phase_np = pred_phase.squeeze().cpu().numpy()
        pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()
        del pred_amp, pred_phase, pred_dp_mag
        torch.cuda.empty_cache()

        # Amplitude normalization INSIDE the support, hide outside as NaN
        # find min/max only over mask
        valid_vals = pred_amp_np[mask]
        if valid_vals.size > 0:
            vmin = valid_vals.min()
            vmax = valid_vals.max()
        else:
            vmin, vmax = 0.0, 1.0

        amp_norm = (pred_amp_np - vmin) / (vmax - vmin + 1e-8)
        amp_norm[~mask] = np.nan  # blank outside support

        # Phase offset using GT amplitude support (already NaN outside)
        phase_offset = deramp(pred_phase_np, mask)

        # Normalize DP magnitude to [0,1]
        dp_norm = pred_dp_np / (pred_dp_np.max() + 1e-8)

        # Amplitude
        im0 = ax[0, col].imshow(amp_norm, cmap='viridis', vmin=0, vmax=1)
        ax[0, col].set_title(f"Run {run_idx} | Chi²={chi_square_list[run_idx]:.4f}", fontsize=25)

        # Phase
        im1 = ax[1, col].imshow(phase_offset, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)

        # DP
        im2 = ax[2, col].imshow(dp_norm, cmap='turbo', vmin=0, vmax=1)

        # apply the same ticks to all 3 rows in this column
        for r in range(3):
            ax[r, col].set_xticks(ticks)
            ax[r, col].set_yticks(ticks)
            ax[r, col].tick_params(labelsize=20)

    # Row labels
    ax[0, 0].set_ylabel("Predicted Amplitude", fontsize=25)
    ax[1, 0].set_ylabel("Predicted Phase (Offset)", fontsize=25)
    ax[2, 0].set_ylabel("Predicted DP Magnitude", fontsize=25)

    # Colorbars
    cbar1 = plt.colorbar(im0, ax=ax[0, -1], fraction=0.046, pad=0.01)
    cbar1.set_label("Amplitude", fontsize=25)
    cbar1.ax.tick_params(labelsize=20)

    cbar2 = plt.colorbar(im1, ax=ax[1, -1], fraction=0.046, pad=0.01)
    cbar2.set_label("Phase (rad)", fontsize=25)
    import math
    cbar2.set_ticks([-math.pi, 0, math.pi])
    cbar2.set_ticklabels(["-π", "0", "π"])
    cbar2.ax.tick_params(labelsize=20)

    cbar3 = plt.colorbar(im2, ax=ax[2, -1], fraction=0.046, pad=0.01)
    cbar3.set_label("Intensity (a.u.)", fontsize=25)
    cbar3.ax.tick_params(labelsize=20)

    plt.suptitle("Top 5 CNN-ViT Predictions Sorted by Chi-square", fontsize=18)
    save_figure(fig, os.path.join(results_dir, "cnnvit_top5_runs_3x5"))

    print("Top-5 reproducibility panel saved at results/cnnvit_top5_runs_3x5.png")

    # 3×3 Plot: Iterative Method vs Model Prediction vs Error Map
    print(f"Reloading Best Run {best_run_idx} for 3x3 comparison...")
    model = build_model(MODEL_PARAMS)
    load_checkpoint(model, best_run_idx)

    with torch.no_grad():
        raw_amp, pred_phase = model(dp_input, fixed_support, prior_amp=prior_amp)
        alpha = get_alpha(final_epoch)  # e.g. 999
        pred_amp = alpha * prior_amp + (1 - alpha) * raw_amp
        #pred_amp, pred_phase = model(dp_input, fixed_support)
        pred_dp_mag = compute_fft_dp(pred_phase, pred_amp)

    # Convert to numpy
    pred_amp_np = pred_amp.squeeze().cpu().numpy()
    pred_phase_np = pred_phase.squeeze().cpu().numpy()
    pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()
    del pred_amp, pred_phase, pred_dp_mag
    torch.cuda.empty_cache()

    # Use iterative recon results
    iter_amp_np = iter_amp.squeeze(0).cpu().numpy()  # already normalized earlier
    iter_phase_np = gt_phase_np  # (iterative phase before offsetting)
    iter_dp_np = iter_dp_tensor.squeeze().cpu().numpy()

    # Support mask for crystal views
    mask = amp_np > 0  # amp_np came from fixed_support earlier (binary-ish)


    # helper: normalize amplitude only inside the support, NaN outside
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


    iter_amp_masked = normalize_amp_inside_mask(iter_amp_np, mask)
    pred_amp_masked = normalize_amp_inside_mask(pred_amp_np, mask)


    # phase offset & masking (outside support = NaN)
    def apply_phase_offset_masked(phase, mask):
        cy, cx = phase.shape[0] // 2, phase.shape[1] // 2
        offset = phase[cy, cx]
        p2 = phase.copy()
        p2[mask] = (p2[mask] - offset + np.pi) % (2 * np.pi) - np.pi
        p2[~mask] = np.nan
        return p2


    iter_phase_offset = apply_phase_offset_masked(iter_phase_np, mask)
    pred_phase_offset = apply_phase_offset_masked(pred_phase_np, mask)

    # DP normalizations
    dp_iter_norm = iter_dp_np / (iter_dp_np.max() + 1e-8)
    dp_pred_norm = pred_dp_np / (pred_dp_np.max() + 1e-8)
    dp_gt_norm = gt_dp_np / (gt_dp_np.max() + 1e-8)

    # differences
    phase_error = (pred_phase_offset - iter_phase_offset + np.pi) % (2 * np.pi) - np.pi
    dp_error = dp_pred_norm - dp_gt_norm
    dp_err_max = np.max(np.abs(dp_error)) + 1e-8  # for symmetric color scale


    # radial profile helper
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


    r_gt = radial_profile(dp_gt_norm)
    r_iter = radial_profile(dp_iter_norm)
    r_pred = radial_profile(dp_pred_norm)

    # ----------------- FIGURE: 3x3 panel -----------------
    # Layout (row x col):
    # [0,0] Iter Amp      [0,1] Iter Phase      [0,2] Iter DP
    # [1,0] ViT Amp       [1,1] ViT Phase       [1,2] ViT DP
    # [2,0] Radial prof   [2,1] DP diff (ML-GT) [2,2] Experimental DP (GT)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    ticks = [0, 16, 32, 48, 64]

    # --- Row 0: Iterative recon ---
    im00 = axes[0, 0].imshow(iter_amp_masked, cmap='viridis', vmin=0, vmax=1)
    axes[0, 0].set_title("Crystal Amplitude (Iterative)", fontsize=20)

    im01 = axes[0, 1].imshow(iter_phase_offset, cmap='twilight_shifted',
                             vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("Crystal Phase (Iterative)", fontsize=20)

    im02 = axes[0, 2].imshow(dp_iter_norm, cmap='turbo', vmin=0, vmax=1)
    axes[0, 2].set_title("DP (Iterative)", fontsize=20)

    # --- Row 1: C-ViT prediction ---
    im10 = axes[1, 0].imshow(pred_amp_masked, cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title("Crystal Amplitude (C-ViT)", fontsize=20)

    im11 = axes[1, 1].imshow(pred_phase_offset, cmap='twilight_shifted',
                             vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("Crystal Phase (C-ViT)", fontsize=20)

    im12 = axes[1, 2].imshow(dp_pred_norm, cmap='turbo', vmin=0, vmax=1)
    axes[1, 2].set_title("DP (C-ViT)", fontsize=20)

    # Row 2, Col 0: Radial profile line plot
    ax_rp = axes[2, 0]
    ax_rp.plot(r_gt, label="GT", lw=2)
    ax_rp.plot(r_iter, label="Iterative", lw=2)
    ax_rp.plot(r_pred, label="C-ViT", lw=2)
    ax_rp.set_yscale('log')
    ax_rp.set_xlabel("Radial Distance (px)", fontsize=15)
    ax_rp.set_ylabel("Intensity (log)", fontsize=15)
    ax_rp.set_title("Radial Intensity Profile of DP", fontsize=20)
    ax_rp.grid(True, linestyle='--', alpha=0.6)
    ax_rp.legend(fontsize=13)

    # Row 2, Col 1: DP difference
    im21 = axes[2, 1].imshow(dp_error, cmap='coolwarm',
                             vmin=-dp_err_max, vmax=dp_err_max)
    axes[2, 1].set_title("DP Difference (C-ViT − GT)", fontsize=20)

    # Row 2, Col 2: Experimental / Ground Truth DP
    im22 = axes[2, 2].imshow(dp_gt_norm, cmap='turbo', vmin=0, vmax=1)
    axes[2, 2].set_title("Experimental DP (GT)", fontsize=20)

    # ticks for all imshow panels
    im_axes = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 1), (2, 2)
    ]
    for (r, c) in im_axes:
        axes[r, c].set_xticks(ticks)
        axes[r, c].set_yticks(ticks)
        axes[r, c].tick_params(labelsize=15)

    # radial profile axis styling (no image ticks fix needed, keep default x ticks)
    ax_rp.tick_params(labelsize=15)

    # colorbars
    # amplitude (iter & vit share same scale)
    cbar00 = plt.colorbar(im00, ax=axes[0, 0], fraction=0.045, pad=0.01)
    cbar00.set_label("Amplitude", fontsize=20)
    cbar00.ax.tick_params(labelsize=20)

    cbar10 = plt.colorbar(im10, ax=axes[1, 0], fraction=0.045, pad=0.01)
    cbar10.set_label("Amplitude", fontsize=20)
    cbar10.ax.tick_params(labelsize=20)

    # phase (iter & vit share same scale, wrap -π..π)
    import math

    cbar01 = plt.colorbar(im01, ax=axes[0, 1], fraction=0.045, pad=0.01)
    cbar01.set_label("Phase (rad)", fontsize=20)
    cbar01.set_ticks([-math.pi, 0, math.pi])
    cbar01.set_ticklabels(["-π", "0", "π"])
    cbar01.ax.tick_params(labelsize=20)

    cbar11 = plt.colorbar(im11, ax=axes[1, 1], fraction=0.045, pad=0.01)
    cbar11.set_label("Phase (rad)", fontsize=15)
    cbar11.set_ticks([-math.pi, 0, math.pi])
    cbar11.set_ticklabels(["-π", "0", "π"])
    cbar11.ax.tick_params(labelsize=20)

    # DP (iter, vit, GT share same 0..1 scale)
    cbar02 = plt.colorbar(im02, ax=axes[0, 2], fraction=0.045, pad=0.01)
    cbar02.set_label("Intensity (a.u.)", fontsize=15)
    cbar02.ax.tick_params(labelsize=20)

    cbar12 = plt.colorbar(im12, ax=axes[1, 2], fraction=0.045, pad=0.01)
    cbar12.set_label("Intensity (a.u.)", fontsize=15)
    cbar12.ax.tick_params(labelsize=20)

    cbar22 = plt.colorbar(im22, ax=axes[2, 2], fraction=0.045, pad=0.01)
    cbar22.set_label("Intensity (a.u.)", fontsize=15)
    cbar22.ax.tick_params(labelsize=20)

    # DP difference colorbar
    cbar21 = plt.colorbar(im21, ax=axes[2, 1], fraction=0.045, pad=0.01)
    cbar21.set_label("Δ Intensity (a.u.)", fontsize=15)
    cbar21.ax.tick_params(labelsize=20)

    plt.suptitle("Phase Retrieval: Iterative vs Learned Model vs Ground Truth", fontsize=18)
    save_figure(fig, os.path.join(results_dir, f"final_3x3_comparison_run_{best_run_idx}"))
    print("3x3 comparison saved: iterative vs model vs GT DP")


    # Extra figure: zoomed 40x40 crystal (best Chi² run)
    from matplotlib import colormaps

    zoom_size = 45

    # pred_amp_masked is 64×64, normalized 0–1 inside support, NaN outside
    H, W = pred_amp_masked.shape
    cy, cx = H // 2, W // 2  # centre of 64×64
    half = zoom_size // 2    # 20 pixels on each side

    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    # Crop amplitude, phase, and mask
    amp_zoom = pred_amp_masked[y0:y1, x0:x1].copy()
    phase_zoom = pred_phase_offset[y0:y1, x0:x1].copy()
    mask_zoom = mask[y0:y1, x0:x1]

    # Outside support to NaN (will be drawn as white)
    amp_zoom[~mask_zoom] = np.nan
    phase_zoom[~mask_zoom] = np.nan

    # Colormaps with NaN to white
    cmap_amp = colormaps["viridis"].copy()
    cmap_phase = colormaps["twilight_shifted"].copy()
    cmap_amp.set_bad("white")
    cmap_phase.set_bad("white")

    # make subfolder like "strong_pc_zoom", "low_p_zoom", etc.
    case_name = os.path.basename(os.path.normpath(output_dir))  # e.g. "strong_pc"
    zoom_dir = os.path.join(results_dir, f"{case_name}_zoom")
    os.makedirs(zoom_dir, exist_ok=True)

    # Amplitude
    fig_a, ax_a = plt.subplots(figsize=(3, 3))
    ax_a.imshow(amp_zoom, cmap=cmap_amp, vmin=0, vmax=1)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(True)  # draw box
    fig_a.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # uses your existing save_figure(fig, name) helper
    save_figure(
        fig_a,
        os.path.join(zoom_dir, "amp"),  # -> amp.png
    )

    # Phase
    fig_p, ax_p = plt.subplots(figsize=(3, 3))
    ax_p.imshow(phase_zoom, cmap=cmap_phase, vmin=-np.pi, vmax=np.pi)
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    for spine in ax_p.spines.values():
        spine.set_visible(True)  # draw box
    fig_p.subplots_adjust(left=0, right=1, bottom=0, top=1)

    save_figure(
        fig_p,
        os.path.join(zoom_dir, "phase"),  # -> phase.png
    )

    # Input DP (64×64)
    # Use the actual input DP to the model: dp_input
    dp_input_np = dp_input.squeeze().cpu().numpy()
    dp_input_np = (dp_input_np - dp_input_np.min()) / (dp_input_np.max() - dp_input_np.min() + 1e-8)

    fig_d, ax_d = plt.subplots(figsize=(3, 3))
    ax_d.imshow(dp_input_np, cmap=colormaps["turbo"], vmin=0, vmax=1)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    for spine in ax_d.spines.values():
        spine.set_visible(True)  # draw box
    fig_d.subplots_adjust(left=0, right=1, bottom=0, top=1)

    save_figure(
        fig_d,
        os.path.join(zoom_dir, "dp_amp"),  # -> dp_amp.png
    )


    def plot_metric_histogram(data, best_idx, xlabel, title, out_path, color):
        """
        Histogram in blue, no grid, no best-run line.
        X-axis is chi² in [0, 3%].
        """
        from matplotlib.ticker import PercentFormatter
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.close(fig)

        # Plot raw chi-square (0–0.03) and show as 0–3%
        ax.hist(data, bins=25, color="#DD8452", edgecolor="black", alpha=0.9)

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Frequency", fontsize=13)
        ax.set_title(title, fontsize=14)

        # 0–3% range
        ax.set_xlim(0.0, 0.02)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))  # 0.01 -> 1%
        ax.tick_params(labelsize=30, width=2.0, length=6)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Thicker box
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)

        # no legend, no grid, no best-run marker
        save_figure(fig, out_path)

    # Chi-square histogram in Sythetic-10_zoom
    plot_metric_histogram(
        chi_square_list,
        best_idx=best_run_idx,  # not used inside, but keeps API
        xlabel="Chi-square",
        title="",
        out_path=os.path.join(zoom_dir, "chi_square_hist"),
        color="#DD8452",
    )


    fig, ax = plt.subplots(figsize=(7, 5))

    # Only GT and C-ViT
    ax.plot(r_gt, label="GT", lw=3, color="#222222")  # neutral dark
    ax.plot(r_iter, label="Iterative", lw=3, color="#4C72B0")  # supportive blue
    ax.plot(r_pred, label="Fourier ViT", lw=3, color="#55A868")  # green
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

    save_figure(fig, os.path.join(zoom_dir, "radial_profile_dp"))

    print(f"Zoomed 40×40 crystal and input DP saved under: {zoom_dir}")


    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_gt, label="GT", lw=2)
    ax.plot(r_iter, label="Iterative", lw=2)
    ax.plot(r_pred, label="C-ViT", lw=2)
    ax.set_yscale('log')
    ax.set_xlabel("Radial Distance (px)", fontsize=13)
    ax.set_ylabel("Intensity (log scale)", fontsize=13)
    ax.set_title("Radial Intensity Profile of DP", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=15)
    save_figure(fig, os.path.join(results_dir, "radial_profile_dp"))

    ##########################################################################
    # --- Top 20 Best Runs by Chi-square (Amplitude + Phase + Save for CM) ---
    ##########################################################################

    print("Generating reproducibility check for Top 20 Chi-square Runs...")

    # Base name for common-mode files (FIX for NameError)
    cm_name = "LCMO-500"

    # Support mask (binary)
    mask = amp_np > 0


    def deramp(phase, mask):
        cy, cx = phase.shape[0] // 2, phase.shape[1] // 2
        offset = phase[cy, cx]
        p2 = phase.copy()
        p2[mask] = (p2[mask] - offset + np.pi) % (2 * np.pi) - np.pi
        p2[~mask] = np.nan
        return p2


    # Best 20 indices
    top_indices = np.argsort(chi_square_list)[:20]

    n_cols = 5
    n_rows = 8  # 4 blocks × (amp, phase)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(24, 32), constrained_layout=True)
    ticks = [0, 16, 32, 48, 64]

    from scipy.io import savemat

    common_mode_dir = os.path.join(results_dir, "common_mode_input")
    os.makedirs(common_mode_dir, exist_ok=True)

    model = build_model(MODEL_PARAMS)

    last_im_amp = None
    last_im_phase = None

    for idx, run_idx in enumerate(top_indices):
        print(f"Run {run_idx} (Chi-square = {chi_square_list[run_idx]:.5f})")
        load_checkpoint(model, run_idx)

        with torch.no_grad():
            raw_amp, pred_phase = model(dp_input, fixed_support, prior_amp=prior_amp)
            alpha = get_alpha(final_epoch)
            pred_amp = alpha * prior_amp + (1 - alpha) * raw_amp

        # to numpy
        pred_amp_np = pred_amp.squeeze().cpu().numpy()
        pred_phase_np = pred_phase.squeeze().cpu().numpy()
        del pred_amp, pred_phase, raw_amp
        torch.cuda.empty_cache()

        # Amplitude normalisation
        vals = pred_amp_np[mask]
        vmin, vmax = (vals.min(), vals.max()) if vals.size else (0.0, 1.0)
        amp_norm = (pred_amp_np - vmin) / (vmax - vmin + 1e-8)
        amp_norm[~mask] = np.nan

        # Phase deramp
        phase_offset = deramp(pred_phase_np, mask)

        # Save amplitude for PyNX common-mode ---
        amp_cm = pred_amp_np * mask
        savemat(
            os.path.join(common_mode_dir, f"{cm_name}-vit-amp{idx}.mat"),
            {"array": amp_cm.astype(np.float32)}
        )
        savemat(
            os.path.join(common_mode_dir, f"{cm_name}-vit-pha{idx}.mat"),
            {"array": phase_offset.astype(np.float32)}
        )

        # Figure placement
        group = idx // n_cols
        col = idx % n_cols
        row_amp = 2 * group
        row_phase = row_amp + 1

        im0 = ax[row_amp, col].imshow(amp_norm, cmap="viridis", vmin=0, vmax=1)
        ax[row_amp, col].set_title(
            f"Run {run_idx} | χ²={chi_square_list[run_idx]:.4f}", fontsize=16
        )

        im1 = ax[row_phase, col].imshow(
            phase_offset, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi
        )

        for r in (row_amp, row_phase):
            ax[r, col].set_xticks(ticks)
            ax[r, col].set_yticks(ticks)
            ax[r, col].tick_params(labelsize=10)

        last_im_amp = im0
        last_im_phase = im1

    # Row labels
    for r in range(0, n_rows, 2):
        ax[r, 0].set_ylabel("Predicted Amplitude", fontsize=18)
        ax[r + 1, 0].set_ylabel("Predicted Phase", fontsize=18)

    # Colorbars
    if last_im_amp is not None:
        cbar_amp = plt.colorbar(last_im_amp, ax=ax[0::2, -1], fraction=0.046, pad=0.01)
        cbar_amp.set_label("Amplitude", fontsize=18)

    if last_im_phase is not None:
        cbar_phase = plt.colorbar(last_im_phase, ax=ax[1::2, -1], fraction=0.046, pad=0.01)
        cbar_phase.set_label("Phase (rad)", fontsize=18)
        cbar_phase.set_ticks([-np.pi, 0, np.pi])
        cbar_phase.set_ticklabels(["-π", "0", "π"])

    plt.suptitle(
        "Top 20 CNN-ViT Crystal Amplitude & Phase (sorted by χ²)",
        fontsize=20
    )
    save_figure(fig, os.path.join(results_dir, "cnnvit_top20_amp_phase"))

    print(f"Saved Top-20 amplitudes for common-mode to: {common_mode_dir}")









