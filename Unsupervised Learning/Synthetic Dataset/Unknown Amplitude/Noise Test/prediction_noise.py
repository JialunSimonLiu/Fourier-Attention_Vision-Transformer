################################################################################
# Fourier ViT prediction file for the synthetic data
# - Jialun Liu, LCN, UCL, 08-10.2025, jialun.liu.17@ucl.ac.uk
################################################################################
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from real_cnnvit_fourier_attention_amp_generalized import *
import os
import tifffile as tiff
os.makedirs("histograms", exist_ok=True)
import matplotlib.ticker as ticker
from matplotlib import cm

# --- Read ENV paths (so run_multi_noise_dp.py can control them)
DP_PATH    = os.getenv("DP_PATH", "gt_dp_mag_1.tif")
CLEAN_PATH   = os.getenv("CLEAN_DP_PATH", "gt_dp_mag_1.tif")
AMP_PATH   = os.getenv("AMP_PATH", "gt_amp_1.tif")
PHASE_PATH = os.getenv("PHASE_PATH", "gt_phase_1.tif")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
HIST_DIR       = os.path.join(OUTPUT_DIR, "histograms")
RESULTS_DIR    = os.path.join(OUTPUT_DIR, "results")
LOSS_DIR       = os.path.join(OUTPUT_DIR, "loss_plots")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)

print(f"[paths] DP_PATH={DP_PATH}")
print(f"[paths] AMP_PATH={AMP_PATH}")
print(f"[paths] PHASE_PATH={PHASE_PATH}")
print(f"[paths] OUTPUT_DIR={OUTPUT_DIR}")

# ============================================================
# Model and Dataset
# ============================================================
NUM_RUNS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PARAMS = {
    "img_size": 64,
    "patch_size": 4,
    "in_channels": 1,
    "embed_dim": 128,
    "num_layers": 10,
    "mlp_dim": 256,
    "dropout": 0.1,
    "use_multiscale": True,
}

def build_model(params: dict) -> VisionTransformer:
    model = VisionTransformer(**params).to(device)
    model.eval()
    return model

def load_checkpoint(model: VisionTransformer, run_id: int) -> None:
    path = os.path.join(CHECKPOINT_DIR, f"single_dp_run_{run_id}.pth")
    model.load_state_dict(torch.load(path, map_location=device))

# ============================================================
# Multi-file Compatible Dataset Loader
# ============================================================
class MultiTifPhaseDataset(Dataset):
    def __init__(self, dp_paths, amp_path, phase_path):
        if isinstance(dp_paths, str):
            dp_paths = [p.strip() for p in dp_paths.split(",")]
        self.dp_paths = dp_paths

        self.amp = tiff.imread(amp_path).astype(np.float32)
        self.amp = self.amp / np.max(self.amp)
        self.amp = torch.tensor(self.amp).unsqueeze(0)

        self.phase = tiff.imread(phase_path).astype(np.float32)
        self.phase = (self.phase / 65535.0) * (2 * np.pi) - np.pi
        self.phase = torch.tensor(self.phase).unsqueeze(0)

    def __len__(self):
        return len(self.dp_paths)

    def __getitem__(self, idx):
        dp = tiff.imread(self.dp_paths[idx]).astype(np.float32)
        dp = dp / np.max(dp)
        dp = torch.tensor(dp).unsqueeze(0)
        return dp, self.phase, self.amp

def get_loader(dp_path=DP_PATH, amp_path=AMP_PATH, phase_path=PHASE_PATH):
    dataset = MultiTifPhaseDataset(dp_path, amp_path, phase_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader

# ============================================================
# FFT, Metrics, and Evaluation
# ============================================================
def compute_fft_dp(phase, fixed_amplitude):
    B, C, H, W = phase.shape
    amplitude_batched = fixed_amplitude.expand(B, 1, H, W)
    density = amplitude_batched * torch.exp(1j * phase)
    dp_mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(density), norm="ortho")))
    dp_mag_norm = dp_mag / (dp_mag.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return dp_mag_norm

def calculate_chi_square(predicted_dp, target_dp):
    predicted_norm = predicted_dp / (predicted_dp.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    target_norm = target_dp / (target_dp.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    chi_square = torch.sum((predicted_norm - target_norm) ** 2, dim=(-1, -2)) / (
        torch.sqrt(torch.sum(predicted_norm ** 2, dim=(-1, -2)) * torch.sum(target_norm ** 2, dim=(-1, -2))) + 1e-8
    )
    return chi_square.item()

def calculate_pcc(predicted_dp, target_dp):
    x = predicted_dp - predicted_dp.mean(dim=(-1, -2), keepdim=True)
    y = target_dp - target_dp.mean(dim=(-1, -2), keepdim=True)
    numerator = (x * y).sum(dim=(-1, -2))
    denominator = torch.sqrt((x ** 2).sum(dim=(-1, -2)) * (y ** 2).sum(dim=(-1, -2)) + 1e-8)
    return (numerator / denominator).item()

# ============================================================
# Main Evaluation Loop
# ============================================================
if __name__ == "__main__":
    test_loader = get_loader()
    dp_mag, phase_gt, fixed_amplitude = next(iter(test_loader))
    dp_mag = dp_mag.to(device)  # 🔹 noisy input DP (model input)
    phase_gt = phase_gt.to(device)
    fixed_amplitude = fixed_amplitude.to(device)

    # 🔹 clean GT DP (for *final* evaluation)
    dp_clean_np = tiff.imread(CLEAN_PATH).astype(np.float32)
    dp_clean_np = dp_clean_np / (dp_clean_np.max() + 1e-8)
    dp_clean = torch.tensor(dp_clean_np).unsqueeze(0).unsqueeze(0).to(device)

    # --- Metrics vs NOISY input DP (dp_mag) ---
    chi_square_list = []  # χ²(noisy)
    pcc_list = []  # PCC(noisy)

    # --- Metrics vs CLEAN DP (dp_clean) ---
    chi_clean_list = []  # χ²(clean)
    pcc_clean_list = []  # PCC(clean)

    for run_id in range(NUM_RUNS):
        print(f"🔎 Evaluating Run {run_id}.")
        model = build_model(MODEL_PARAMS)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"single_dp_run_{run_id}.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        with torch.no_grad():
            pred_amp, pred_phase = model(dp_mag, fixed_amplitude)
            pred_dp_mag = compute_fft_dp(pred_phase, pred_amp)

            # --- 1) Metrics w.r.t. NOISY input DP (selection criterion) ---
            chi_noisy = calculate_chi_square(pred_dp_mag, dp_mag)
            pcc_noisy = calculate_pcc(pred_dp_mag, dp_mag)

            # --- 2) Metrics w.r.t. CLEAN DP (for reporting) ---
            chi_clean = calculate_chi_square(pred_dp_mag, dp_clean)
            pcc_clean = calculate_pcc(pred_dp_mag, dp_clean)

        # store noisy metrics
        chi_square_list.append(chi_noisy)
        pcc_list.append(pcc_noisy)

        # store clean metrics
        chi_clean_list.append(chi_clean)
        pcc_clean_list.append(pcc_clean)

    chi_square_list = np.array(chi_square_list)  # χ² vs noisy DP
    pcc_list = np.array(pcc_list)  # PCC vs noisy DP
    chi_clean_list = np.array(chi_clean_list)  # χ² vs clean DP
    pcc_clean_list = np.array(pcc_clean_list)  # PCC vs clean DP

    # 🔹 SELECTION: best/worst by χ² vs NOISY input DP
    best_run_idx = np.argmin(chi_square_list)
    worst_run_idx = np.argmax(chi_square_list)

    print("\nBest Run (selected vs *noisy* input DP):", best_run_idx)
    print("  Chi-square (noisy DP):", chi_square_list[best_run_idx])
    print("  PCC        (noisy DP):", pcc_list[best_run_idx])
    print("  Chi-square (CLEAN DP):", chi_clean_list[best_run_idx])
    print("  PCC        (CLEAN DP):", pcc_clean_list[best_run_idx])

    print("\nWorst Run (by χ² vs noisy input DP):", worst_run_idx)
    print("  Chi-square (noisy DP):", chi_square_list[worst_run_idx])
    print("  PCC        (noisy DP):", pcc_list[worst_run_idx])
    print("  Chi-square (CLEAN DP):", chi_clean_list[worst_run_idx])
    print("  PCC        (CLEAN DP):", pcc_clean_list[worst_run_idx])

    from pathlib import Path

    def save_figure(fig, path, dpi=300, bbox_inches='tight'):
        p = Path(path).with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)

    # --- Chi-square histogram (now vs NOISY DP) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(chi_square_list, bins=50, color="#4C72B0", edgecolor='black', alpha=0.8)
    ax.axvline(chi_square_list[best_run_idx], color='red', linestyle='--',
               label=f"Best Run {best_run_idx}")
    ax.set_xlabel("Chi-square (vs noisy DP)", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title("Distribution of Chi-square Across Runs", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=20)
    save_figure(fig, os.path.join(HIST_DIR, "chi_square_histogram"))

    # --- PCC histogram (now vs NOISY DP) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pcc_list, bins=50, color="#55A868", edgecolor='black', alpha=0.8)
    ax.axvline(pcc_list[best_run_idx], color='red', linestyle='--',
               label=f"Best Run {best_run_idx}")
    ax.set_xlabel("Pearson Correlation Coefficient (vs noisy DP)", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title("Distribution of PCC Across Runs", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=20)
    save_figure(fig, os.path.join(HIST_DIR, "pcc_histogram"))

    # --- Best loss curve (unchanged) ---
    loss_history = np.load(os.path.join(LOSS_DIR, f"loss_per_epoch_run_{best_run_idx}.npy"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history,
            marker='o', linestyle='-', color="#C44E52")
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Loss (log scale)", fontsize=20)
    ax.set_title(f"Loss Curve of Best Run ({best_run_idx})", fontsize=20)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=20)
    save_figure(fig, os.path.join(LOSS_DIR, f"best_loss_curve_run_{best_run_idx}"))

    # --- Predict again using Best Run and Plot Comparison ---
    print(f"Generating final comparison figure for Best Run {best_run_idx}...")
    model = build_model(MODEL_PARAMS)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"single_dp_run_{best_run_idx}.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    model_worst = build_model(MODEL_PARAMS)
    checkpoint_path_worst = os.path.join(CHECKPOINT_DIR, f"single_dp_run_{worst_run_idx}.pth")
    model_worst.load_state_dict(torch.load(checkpoint_path_worst, map_location=device))
    model_worst.eval()

    with torch.no_grad():
        pred_amp, pred_phase = model(dp_mag, fixed_amplitude)
        pred_dp_mag = compute_fft_dp(pred_phase, pred_amp)
        pred_amp_worst, pred_phase_worst = model_worst(dp_mag, fixed_amplitude)
        pred_dp_mag_worst = compute_fft_dp(pred_phase_worst, pred_amp_worst)
        true_dp_mag = dp_mag

    # Move tensors to CPU and numpy
    pred_amp_np = pred_amp.squeeze().cpu().numpy()
    pred_amp_norm = (pred_amp_np - pred_amp_np.min()) / (pred_amp_np.max() - pred_amp_np.min() + 1e-8)
    gt_phase_np = phase_gt.squeeze().cpu().numpy()
    pred_phase_np = pred_phase.squeeze().cpu().numpy()
    gt_dp_np = true_dp_mag.squeeze().cpu().numpy()
    pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()
    amp_np = fixed_amplitude.squeeze().cpu().numpy()
    pred_amp_worst_np = pred_amp_worst.squeeze().cpu().numpy()
    pred_phase_worst_np = pred_phase_worst.squeeze().cpu().numpy()
    pred_dp_worst_np = pred_dp_mag_worst.squeeze().cpu().numpy()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Plot original classic 2x3 comparison (best) ---
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    im0 = ax[0, 0].imshow(amp_np, cmap='viridis')
    ax[0, 0].set_title("Ground Truth Amplitude")
    ax[0, 0].axis('off')
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(gt_phase_np, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    ax[0, 1].set_title("Ground Truth Phase")
    ax[0, 1].axis('off')
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[0, 2].imshow(gt_dp_np / np.max(gt_dp_np), cmap='jet')
    ax[0, 2].set_title("Ground Truth DP Magnitude")
    ax[0, 2].axis('off')
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    im3 = ax[1, 0].imshow(pred_amp_np, cmap='viridis')
    ax[1, 0].set_title("Predicted Amplitude")
    ax[1, 0].axis('off')
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im4 = ax[1, 1].imshow(pred_phase_np, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    ax[1, 1].set_title("Predicted Phase")
    ax[1, 1].axis('off')
    plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im5 = ax[1, 2].imshow(pred_dp_np / np.max(pred_dp_np), cmap='jet')
    ax[1, 2].set_title("Predicted DP Magnitude")
    ax[1, 2].axis('off')
    plt.colorbar(im5, ax=ax[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"final_comparison_best_run_{best_run_idx}.png"), dpi=300)
    plt.close()

    # --- Plot original classic 2x3 comparison (worst) ---
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    im0 = ax[0, 0].imshow(amp_np, cmap='viridis')
    ax[0, 0].set_title("Ground Truth Amplitude")
    ax[0, 0].axis('off')
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(gt_phase_np, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    ax[0, 1].set_title("Ground Truth Phase")
    ax[0, 1].axis('off')
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[0, 2].imshow(gt_dp_np / np.max(gt_dp_np), cmap='jet')
    ax[0, 2].set_title("Ground Truth DP Magnitude")
    ax[0, 2].axis('off')
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    im3 = ax[1, 0].imshow(pred_amp_worst_np, cmap='viridis')
    ax[1, 0].set_title("Predicted Amplitude")
    ax[1, 0].axis('off')
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im4 = ax[1, 1].imshow(pred_phase_worst_np, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    ax[1, 1].set_title("Predicted Phase")
    ax[1, 1].axis('off')
    plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im5 = ax[1, 2].imshow(pred_dp_worst_np / np.max(pred_dp_worst_np), cmap='jet')
    ax[1, 2].set_title("Predicted DP Magnitude")
    ax[1, 2].axis('off')
    plt.colorbar(im5, ax=ax[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"final_comparison_worst_run_{worst_run_idx}.png"), dpi=300)
    plt.close()

    print(f"Final classic comparison saved at {os.path.join(RESULTS_DIR, f'final_comparison_best_run_{best_run_idx}.png')}")

    # Top 5 Best Runs by Chi-square
    print("Generating reproducibility check for Top 5 Chi-square Runs...")

    amp_np = fixed_amplitude.squeeze().cpu().numpy()
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


    # --- Sort and record (top-5 by χ² vs NOISY DP) ---
    top5_indices = np.argsort(chi_square_list)[:5]
    top5_txt_path = os.path.join(RESULTS_DIR, "top5_chisq.txt")
    with open(top5_txt_path, "w") as f:
        f.write("Top 5 Runs by Chi-square (lower is better)\n")
        f.write("rank\trun_id\tchi2_noisy\tpcc_noisy\tchi2_clean\tpcc_clean\n")
        for rank, run_idx in enumerate(top5_indices, start=1):
            f.write(
                f"{rank}\t{int(run_idx)}\t"
                f"{float(chi_square_list[run_idx]):.8f}\t"  # χ² vs noisy
                f"{float(pcc_list[run_idx]):.8f}\t"  # PCC vs noisy
                f"{float(chi_clean_list[run_idx]):.8f}\t"  # χ² vs clean
                f"{float(pcc_clean_list[run_idx]):.8f}\n"  # PCC vs clean
            )

    # --- Plot grid ---
    fig, ax = plt.subplots(3, 5, figsize=(25, 15), constrained_layout=True)
    ticks = [0, 16, 32, 48, 64]
    model = build_model(MODEL_PARAMS)

    for col, run_idx in enumerate(top5_indices):
        print(f"Run {run_idx} (Chi-square = {chi_square_list[run_idx]:.5f})")
        load_checkpoint(model, run_idx)

        with torch.no_grad():
            pred_amp, pred_phase = model(dp_mag, fixed_amplitude)
            pred_dp_mag = compute_fft_dp(pred_phase, pred_amp)

        pred_amp_np = pred_amp.squeeze().cpu().numpy()
        pred_phase_np = pred_phase.squeeze().cpu().numpy()
        pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()
        del pred_amp, pred_phase, pred_dp_mag
        torch.cuda.empty_cache()

        # Amplitude normalization inside mask
        valid_vals = pred_amp_np[mask]
        if valid_vals.size > 0:
            vmin, vmax = valid_vals.min(), valid_vals.max()
        else:
            vmin, vmax = 0.0, 1.0
        amp_norm = (pred_amp_np - vmin) / (vmax - vmin + 1e-8)
        amp_norm[~mask] = np.nan

        # Phase offset & DP normalization
        phase_offset = deramp(pred_phase_np, mask)
        dp_norm = pred_dp_np / (pred_dp_np.max() + 1e-8)

        # Row 1–3 plotting
        im0 = ax[0, col].imshow(amp_norm, cmap="viridis", vmin=0, vmax=1)
        ax[0, col].set_title(f"Run {run_idx} | Chi²={chi_square_list[run_idx]:.4f}", fontsize=25)
        im1 = ax[1, col].imshow(phase_offset, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
        im2 = ax[2, col].imshow(dp_norm, cmap="turbo", vmin=0, vmax=1)

        for r in range(3):
            ax[r, col].set_xticks(ticks)
            ax[r, col].set_yticks(ticks)
            ax[r, col].tick_params(labelsize=20)

    # --- Row labels ---
    ax[0, 0].set_ylabel("Predicted Amplitude", fontsize=25)
    ax[1, 0].set_ylabel("Predicted Phase (Offset)", fontsize=25)
    ax[2, 0].set_ylabel("Predicted DP Magnitude", fontsize=25)

    # --- Shared colorbars (once only) ---
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

    # --- Final ---
    plt.suptitle("Top 5 CNN-ViT Predictions Sorted by Chi-square", fontsize=18)
    save_figure(fig, os.path.join(RESULTS_DIR, "cnnvit_top5_runs_3x5"), dpi=300, bbox_inches="tight")
    print(f"Top-5 reproducibility panel saved at {os.path.join(RESULTS_DIR, 'cnnvit_top5_runs_3x5.png')}")

    # --- 3×3 Phase offset + error maps ---
    print(f"Generating Phase offset + error maps figure for Best Run {best_run_idx}...")

    model = build_model(MODEL_PARAMS)
    load_checkpoint(model, best_run_idx)

    with torch.no_grad():
        pred_amp, pred_phase = model(dp_mag, fixed_amplitude)
        pred_dp_mag = compute_fft_dp(pred_phase, pred_amp)

    pred_phase_np = pred_phase.squeeze().cpu().numpy()
    pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()
    pred_amp_np = pred_amp.squeeze().cpu().numpy()

    mask = amp_np > 0
    ticks = [0, 16, 32, 48, 64]
    font_size = 11

    gt_phase_deramped = deramp(phase_gt.squeeze().cpu().numpy(), mask)
    pred_phase_deramped = deramp(pred_phase_np, mask)
    pred_amp_norm = (pred_amp_np - pred_amp_np.min()) / (pred_amp_np.max() - pred_amp_np.min() + 1e-8)

    phase_error = (pred_phase_deramped - gt_phase_deramped + np.pi) % (2 * np.pi) - np.pi
    dp_error = (pred_dp_np / np.max(pred_dp_np)) - (gt_dp_np / np.max(gt_dp_np))

    amp_norm = (amp_np - amp_np.min()) / (amp_np.max() - amp_np.min() + 1e-8)
    dp_gt_norm = (gt_dp_np - gt_dp_np.min()) / (gt_dp_np.max() - gt_dp_np.min() + 1e-8)
    dp_pr_norm = (pred_dp_np - pred_dp_np.min()) / (pred_dp_np.max() - pred_dp_np.min() + 1e-8)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    titles = [
        'GT Amplitude', 'GT Phase (Phase Offset)', 'GT DP Magnitude',
        'Predicted Amplitude', 'Predicted Phase (Phase Offset)', 'Predicted DP Magnitude',
        'Radial Profile of DP (log)', 'DP Error (Pred - GT)', 'Phase Scatter Plot'
    ]
    colormaps = [
        'viridis', 'twilight_shifted', 'turbo',
        'viridis', 'twilight_shifted', 'turbo',
        None, 'coolwarm', None
    ]
    vmins = [0, -np.pi, 0, 0, -np.pi, 0, None, -np.max(np.abs(dp_error)), None]
    vmaxs = [1, np.pi, 1, 1, np.pi, 1, None, np.max(np.abs(dp_error)), None]
    images = [
        amp_norm, gt_phase_deramped, dp_gt_norm,
        pred_amp_norm, pred_phase_deramped, dp_pr_norm,
        None, dp_error, None
    ]

    def radial_profile(img, center=None):
        y, x = np.indices(img.shape)
        if center is None:
            center = (img.shape[0] // 2, img.shape[1] // 2)
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(np.int32)
        tbin = np.bincount(r.ravel(), img.ravel())
        nr = np.bincount(r.ravel())
        return tbin / (nr + 1e-8)

    r_gt = radial_profile(dp_clean_np)
    r_pred = radial_profile(dp_pr_norm)

    for idx, ax in enumerate(axes.flat):
        if idx == 6:  # radial profile
            ax.plot(r_gt, label="GT", linewidth=2)
            ax.plot(r_pred, label="Pred", linewidth=2)
            ax.set_yscale('log')
            ax.set_title(titles[idx], fontsize=20)
            ax.set_xlabel("Radial Q (px)", fontsize=20)
            ax.set_ylabel("Intensity", fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=20)
            continue
        elif idx == 8:  # scatter
            gt_flat = gt_phase_deramped[mask].flatten()
            pred_flat = pred_phase_deramped[mask].flatten()
            coeffs = np.polyfit(gt_flat, pred_flat, 1)
            fit_line = np.poly1d(coeffs)(gt_flat)
            residuals = pred_flat - gt_flat
            std_dev = np.std(residuals)
            mae = np.mean(np.abs(residuals))
            pcc = np.corrcoef(gt_flat, pred_flat)[0, 1]
            ax.scatter(gt_flat, pred_flat, s=30, alpha=0.3, color='#4C72B0', label="Data")
            ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', label="Ideal", linewidth=2.0)
            ax.plot(np.sort(gt_flat), np.poly1d(coeffs)(np.sort(gt_flat)), 'b-', label="Fit", linewidth=1.5)
            ax.fill_between(np.sort(gt_flat),
                            np.poly1d(coeffs)(np.sort(gt_flat)) - std_dev,
                            np.poly1d(coeffs)(np.sort(gt_flat)) + std_dev,
                            color='blue', alpha=0.2, label="±1σ")
            ax.set_xlim([-np.pi, np.pi]); ax.set_ylim([-np.pi, np.pi])
            ax.set_xlabel("Ground Truth Phase (rad)", fontsize=20)
            ax.set_ylabel("Predicted Phase (rad)", fontsize=20)
            ax.set_title("Phase Scatter Plot", fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=15, loc="upper left")
            print(f"  Pearson R={pcc:.4f}, MAE={mae:.4f}, σ={std_dev:.4f}")
        else:
            im = ax.imshow(images[idx], cmap=colormaps[idx], vmin=vmins[idx], vmax=vmaxs[idx])
            ax.set_title(titles[idx], fontsize=20)
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.01)
            if 'Phase' in titles[idx] or 'Error' in titles[idx]:
                cbar.set_label("Phase (rad)", fontsize=20)
            elif 'Amplitude' in titles[idx]:
                cbar.set_label("Amplitude", fontsize=20)
            elif 'DP' in titles[idx]:
                cbar.set_label("Intensity (a.u.)", fontsize=20)

    plt.suptitle(f"Best Run {best_run_idx}: Phase Retrieval Evaluation", fontsize=20)
    save_figure(fig, os.path.join(RESULTS_DIR, f"final_3x3_comparison_run_{best_run_idx}"), dpi=300, bbox_inches='tight')
    print(f"Final 3x3 comparison saved at {os.path.join(RESULTS_DIR, f'final_3x3_comparison_run_{best_run_idx}.png')}")

    # ============================================================
    # Extra figure: zoomed 40x40 crystal (best Chi² run)
    # ============================================================
    import numpy as np
    from matplotlib import colormaps

    zoom_size = 40
    H, W = pred_amp_norm.shape
    cy, cx = H // 2, W // 2  # centre of 64x64
    half = zoom_size // 2  # 20 pixels

    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    # Crop amplitude, phase, and mask
    amp_zoom = pred_amp_norm[y0:y1, x0:x1].copy()
    phase_zoom = pred_phase_deramped[y0:y1, x0:x1].copy()
    mask_zoom = mask[y0:y1, x0:x1]

    # Outside support -> NaN (will be drawn as white)
    amp_zoom[~mask_zoom] = np.nan
    phase_zoom[~mask_zoom] = np.nan

    # Colormaps with NaN -> white
    cmap_amp = colormaps["viridis"].copy()
    cmap_phase = colormaps["twilight_shifted"].copy()
    cmap_amp.set_bad("white")
    cmap_phase.set_bad("white")

    # --- make subfolder like "strong_pc_zoom", "low_p_zoom", etc. ---
    case_name = os.path.basename(os.path.normpath(OUTPUT_DIR))  # e.g. "strong_pc"
    zoom_dir = os.path.join(RESULTS_DIR, f"{case_name}_zoom")
    os.makedirs(zoom_dir, exist_ok=True)

    # --------- Amplitude-only PNG: amp.png ---------
    fig_a, ax_a = plt.subplots(figsize=(3, 3))
    ax_a.imshow(amp_zoom, cmap=cmap_amp, vmin=0, vmax=1)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(True)  # draw box
    fig_a.subplots_adjust(left=0, right=1, bottom=0, top=1)

    save_figure(
        fig_a,
        os.path.join(zoom_dir, "amp"),  # -> amp.png
        dpi=300,
        bbox_inches="tight",
    )

    # --------- Phase-only PNG: phase.png ---------
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
        dpi=300,
        bbox_inches="tight",
    )

    # --------- Input DP (64x64), no words: dp_amp.png ---------
    # Use the actual input DP to the model: dp_mag
    dp_input_np = dp_mag.squeeze().cpu().numpy()
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
        dpi=300,
        bbox_inches="tight",
    )

    # --- Multi-run Metric Histogram Panel ---
    print("Generating multi-run histogram panel...")
    from skimage.metrics import structural_similarity as ssim

    def circular_chi_square(pred_np, gt_np, mask):
        diff = (pred_np - gt_np + np.pi) % (2 * np.pi) - np.pi
        return np.mean((diff[mask]) ** 2)

    pcc_phase_all, chisq_phase_all, ssim_phase_all = [], [], []
    gt_np = phase_gt.squeeze().cpu().numpy()
    gt_phase_offset = apply_phase_offset(gt_np, mask)

    for run_id in range(NUM_RUNS):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"single_dp_run_{run_id}.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        with torch.no_grad():
            pred_amp_run, pred_phase_run = model(dp_mag, fixed_amplitude)
            _ = compute_fft_dp(pred_phase_run, pred_amp_run)
        pred_np = pred_phase_run.squeeze().cpu().numpy()
        pred_offset = apply_phase_offset(pred_np, mask)

        valid_pred = pred_offset[mask]; valid_gt = gt_phase_offset[mask]
        pred_c = valid_pred - np.mean(valid_pred); gt_c = valid_gt - np.mean(valid_gt)
        pcc_val = np.sum(pred_c * gt_c) / (np.sqrt(np.sum(pred_c ** 2) * np.sum(gt_c ** 2)) + 1e-8)
        chisq_val = circular_chi_square(pred_offset, gt_phase_offset, mask)

        y, x = np.where(mask); y0, y1 = y.min(), y.max(); x0, x1 = x.min(), x.max()
        crop_pred = pred_offset[y0:y1 + 1, x0:x1 + 1]; crop_gt = gt_phase_offset[y0:y1 + 1, x0:x1 + 1]
        crop_mask = mask[y0:y1 + 1, x0:x1 + 1]
        ssim_val = ssim(crop_pred, crop_gt, data_range=2 * np.pi, win_size=11,
                        gaussian_weights=True, use_sample_covariance=False, mask=crop_mask)

        pcc_phase_all.append(pcc_val); chisq_phase_all.append(chisq_val); ssim_phase_all.append(ssim_val)

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    axes[0, 0].hist(chi_square_list, bins=50, color='#4C72B0', edgecolor='black', alpha=0.85)
    axes[0, 1].hist(pcc_list, bins=50, color='#D55E00', edgecolor='black', alpha=0.85)
    axes[0, 2].hist(ssim_phase_all, bins=50, color='#55A868', edgecolor='black', alpha=0.85)
    axes[1, 0].hist(chisq_phase_all, bins=50, color='#937860', edgecolor='black', alpha=0.85)
    axes[1, 1].hist(pcc_phase_all, bins=50, color='#C44E52', edgecolor='black', alpha=0.85)
    axes[1, 2].axis('off')
    plt.suptitle("Metric Distributions Across 30 Runs", fontsize=22)
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_histogram_panel_ssim.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(RESULTS_DIR, 'metrics_histogram_panel_ssim.png')}")
