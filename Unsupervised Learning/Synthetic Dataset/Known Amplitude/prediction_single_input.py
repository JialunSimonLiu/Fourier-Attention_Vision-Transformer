import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from real_cnnvit_fourier_attention_generalized import *
import os
import tifffile as tiff
os.makedirs("histograms", exist_ok=True)
import matplotlib.ticker as ticker
from matplotlib import cm

NUM_RUNS = 100
CHECKPOINT_DIR = "checkpoints"
HIST_DIR = "histograms"
RESULTS_DIR = "results"
LOSS_DIR = "loss_plots"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model Parameters
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

os.makedirs("results", exist_ok=True)
# =============================================================================
# Model helpers
# =============================================================================
def build_model(params: dict) -> VisionTransformer:
    """Instantiate a VisionTransformer with given parameters."""
    model = VisionTransformer(**params).to(device)
    model.eval()
    return model

def load_checkpoint(model: VisionTransformer, run_id: int) -> None:
    """Load state for a specific run into the model."""
    path = os.path.join(CHECKPOINT_DIR, f"single_dp_run_{run_id}.pth")
    model.load_state_dict(torch.load(path, map_location=device))

# Dataset class
class SingleTifPhaseDataset(Dataset):
    def __init__(self, dp_path, amp_path, phase_path):
        self.dp = tiff.imread(dp_path).astype(np.float32)
        self.dp = self.dp / np.max(self.dp)
        self.dp = torch.tensor(self.dp).unsqueeze(0)

        self.amp = tiff.imread(amp_path).astype(np.float32)
        self.amp = self.amp / np.max(self.amp)
        self.amp = torch.tensor(self.amp).unsqueeze(0)

        self.phase = tiff.imread(phase_path).astype(np.float32)
        self.phase = (self.phase / 65535.0) * (2 * np.pi) - np.pi
        self.phase = torch.tensor(self.phase).unsqueeze(0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.dp, self.phase, self.amp

# DataLoader
def get_single_tif_loader(dp_path="gt_dp_mag_1.tif",
                          amp_path="gt_amp_1.tif",
                          phase_path="gt_phase_1.tif"):
    dataset = SingleTifPhaseDataset(dp_path, amp_path, phase_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader

# Fourier Transform for DP Reconstruction
def compute_fft_dp(phase, fixed_amplitude):
    B, C, H, W = phase.shape
    amplitude_batched = fixed_amplitude.expand(B, 1, H, W)
    density = amplitude_batched * torch.exp(1j * phase)
    dp_mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(density))))
    dp_mag_norm = dp_mag / (dp_mag.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return dp_mag_norm

# Metric functions
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

# ---------------------- 🔹 MAIN EVALUATION 🔹 ----------------------
if __name__ == "__main__":
    test_loader = get_single_tif_loader()

    dp_mag, phase_gt, fixed_amplitude = next(iter(test_loader))
    dp_mag = dp_mag.to(device)
    phase_gt = phase_gt.to(device)
    fixed_amplitude = fixed_amplitude.to(device)

    chi_square_list = []
    pcc_list = []

    for run_id in range(NUM_RUNS):
        print(f"🔎 Evaluating Run {run_id}...")
        model = build_model(MODEL_PARAMS)
        checkpoint_path = f"checkpoints/single_dp_run_{run_id}.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        with torch.no_grad():
            pred_phase = model(dp_mag, fixed_amplitude)
            pred_dp_mag = compute_fft_dp(pred_phase, fixed_amplitude)
            true_dp_mag = dp_mag

            chi = calculate_chi_square(pred_dp_mag, true_dp_mag)
            pcc = calculate_pcc(pred_dp_mag, true_dp_mag)

            chi_square_list.append(chi)
            pcc_list.append(pcc)

    chi_square_list = np.array(chi_square_list)
    pcc_list = np.array(pcc_list)

    best_run_idx = np.argmin(chi_square_list)
    worst_run_idx = np.argmax(chi_square_list)

    print("\nBest Run:", best_run_idx)
    print("Best Chi-square:", chi_square_list[best_run_idx])
    print("PCC of Best Run:", pcc_list[best_run_idx])

    print("\nWorst Run:", worst_run_idx)
    print("Worst Chi-square:", chi_square_list[worst_run_idx])
    print("PCC of Worst Run:", pcc_list[worst_run_idx])


    def save_figure(fig, path, dpi=300, bbox_inches='tight'):
        fig.savefig(f"{path}.png", dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)


    # --- Upgraded: Chi-square histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(chi_square_list, bins=50, color="#4C72B0", edgecolor='black', alpha=0.8)
    ax.axvline(chi_square_list[best_run_idx], color='red', linestyle='--', label=f"Best Run {best_run_idx}")
    ax.set_xlabel("Chi-square", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title("Distribution of Chi-square Across Runs", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=20)
    save_figure(fig, "histograms/chi_square_histogram")

    # --- Upgraded: PCC histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pcc_list, bins=50, color="#55A868", edgecolor='black', alpha=0.8)
    ax.axvline(pcc_list[best_run_idx], color='red', linestyle='--', label=f"Best Run {best_run_idx}")
    ax.set_xlabel("Pearson Correlation Coefficient", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title("Distribution of PCC Across Runs", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=20)
    save_figure(fig, "histograms/pcc_histogram")

    # --- Upgraded: Best loss curve ---
    loss_history = np.load(f"loss_plots/loss_per_epoch_run_{best_run_idx}.npy")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color="#C44E52")
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Loss (log scale)", fontsize=20)
    ax.set_title(f"Loss Curve of Best Run ({best_run_idx})", fontsize=20)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=20)
    save_figure(fig, f"loss_plots/best_loss_curve_run_{best_run_idx}")

    # --- Predict again using Best Run and Plot Comparison ---
    print(f"📸 Generating final comparison figure for Best Run {best_run_idx}...")
    model = build_model(MODEL_PARAMS)
    checkpoint_path = f"checkpoints/single_dp_run_{best_run_idx}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    model_worst = build_model(MODEL_PARAMS)
    checkpoint_path_worst = f"checkpoints/single_dp_run_{worst_run_idx}.pth"
    model_worst.load_state_dict(torch.load(checkpoint_path_worst, map_location=device))
    model_worst.eval()

    with torch.no_grad():
        pred_phase = model(dp_mag, fixed_amplitude)
        pred_dp_mag = compute_fft_dp(pred_phase, fixed_amplitude)
        pred_phase_worst = model_worst(dp_mag, fixed_amplitude)
        pred_dp_mag_worst = compute_fft_dp(pred_phase_worst, fixed_amplitude)
        true_dp_mag = dp_mag

    # Move tensors to CPU and numpy
    gt_phase_np = phase_gt.squeeze().cpu().numpy()
    pred_phase_np = pred_phase.squeeze().cpu().numpy()
    gt_dp_np = true_dp_mag.squeeze().cpu().numpy()
    pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()
    amp_np = fixed_amplitude.squeeze().cpu().numpy()
    pred_phase_worst_np = pred_phase_worst.squeeze().cpu().numpy()
    pred_dp_worst_np = pred_dp_mag_worst.squeeze().cpu().numpy()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # --- Plot original classic 2x3 comparison ---
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

    im3 = ax[1, 0].imshow(amp_np, cmap='viridis')
    ax[1, 0].set_title("Predicted Amplitude (Fixed)")
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
    plt.savefig(f"results/final_comparison_best_run_{best_run_idx}.png", dpi=300)
    plt.close()

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

    im3 = ax[1, 0].imshow(amp_np, cmap='viridis')
    ax[1, 0].set_title("Predicted Amplitude (Fixed)")
    ax[1, 0].axis('off')
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im4 = ax[1, 1].imshow(pred_phase_worst_np, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    ax[1, 1].set_title("Predicted Phase")
    ax[1, 1].axis('off')
    plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im5 = ax[1, 2].imshow(pred_dp_worst_np / np.max(pred_dp_np), cmap='jet')
    ax[1, 2].set_title("Predicted DP Magnitude")
    ax[1, 2].axis('off')
    plt.colorbar(im5, ax=ax[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"results/final_comparison_worst_run_{worst_run_idx}.png", dpi=300)
    plt.close()


    print(f"✅ Final classic comparison saved at results/final_comparison_best_run_{best_run_idx}.png")

    # --- 🔝 Top 5 Best Runs by Chi-square ---
    print("📸 Generating reproducibility check for Top 5 Chi-square Runs...")

    # Use ground truth amplitude mask for all phase offsetting
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

    top5_indices = np.argsort(chi_square_list)[:5]
    fig, ax = plt.subplots(3, 5, figsize=(25, 15), constrained_layout=True)
    ticks = [0, 16, 32, 48, 64]

    model = build_model(MODEL_PARAMS)

    for col, run_idx in enumerate(top5_indices):
        print(f"Run {run_idx} (Chi-square = {chi_square_list[run_idx]:.5f})")
        load_checkpoint(model, run_idx)

        with torch.no_grad():
            pred_phase = model(dp_mag, fixed_amplitude)
            pred_dp_mag = compute_fft_dp(pred_phase, fixed_amplitude)

        # Convert to numpy
        pred_phase_np = pred_phase.squeeze().cpu().numpy()
        pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()

        # Normalize amplitude
        fix_amp_masked = amp_np.copy()
        fix_amp_masked[~mask] = 0  # force background zero
        valid_values = fix_amp_masked[mask]
        amp_norm = np.zeros_like(fix_amp_masked)
        amp_norm[mask] = (valid_values - valid_values.min()) / (valid_values.max() - valid_values.min() + 1e-8)

        # Phase offset using GT amplitude support
        phase_offset = deramp(pred_phase_np, mask)

        # Normalize DP
        dp_norm = pred_dp_np / (pred_dp_np.max() + 1e-8)

        # --- Row 1: Amplitude ---
        im0 = ax[0, col].imshow(amp_norm, cmap='viridis', vmin=0, vmax=1)
        ax[0, col].set_title(f"Run {run_idx} | Chi²={chi_square_list[run_idx]:.4f}", fontsize=25)
        ax[0, col].tick_params(labelsize=20)

        # --- Row 2: Phase ---
        im1 = ax[1, col].imshow(phase_offset, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
        ax[1, col].tick_params(labelsize=20)

        # --- Row 3: DP ---
        im2 = ax[2, col].imshow(dp_norm, cmap='turbo', vmin=0, vmax=1)
        ax[2, col].tick_params(labelsize=20)

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
    cbar2.ax.tick_params(labelsize=20)
    cbar3 = plt.colorbar(im2, ax=ax[2, -1], fraction=0.046, pad=0.01)
    cbar3.set_label("Intensity (a.u.)", fontsize=25)
    cbar3.ax.tick_params(labelsize=20)

    plt.suptitle("Top 5 CNN-ViT Predictions Sorted by Chi-square", fontsize=18)
    save_figure(fig, os.path.join(RESULTS_DIR, "cnnvit_top5_runs_3x5"), dpi=300, bbox_inches='tight')

    print("✅ Top-5 reproducibility panel saved at results/cnnvit_top5_runs_3x5.png")

    # --- Now do deramping and advanced 3x3 plotting ---
    print(f"📸 Generating Phase offset + error maps figure for Best Run {best_run_idx}...")

    # Reload best model checkpoint to ensure consistency
    model = build_model(MODEL_PARAMS)
    load_checkpoint(model, best_run_idx)

    with torch.no_grad():
        pred_phase = model(dp_mag, fixed_amplitude)
        pred_dp_mag = compute_fft_dp(pred_phase, fixed_amplitude)

    # Convert predictions to numpy
    pred_phase_np = pred_phase.squeeze().cpu().numpy()
    pred_dp_np = pred_dp_mag.squeeze().cpu().numpy()

    # Use support mask for deramping and normalization
    mask = amp_np > 0

    from matplotlib.colors import Normalize
    import matplotlib.patches as mpatches

    # Tick labels for axis
    ticks = [0, 16, 32, 48, 64]
    font_size = 11

    # Prepare deramped and difference maps
    gt_phase_deramped = deramp(gt_phase_np, mask)
    pred_phase_deramped = deramp(pred_phase_np, mask)

    phase_error = (pred_phase_deramped - gt_phase_deramped + np.pi) % (2 * np.pi) - np.pi
    dp_error = (pred_dp_np / np.max(pred_dp_np)) - (gt_dp_np / np.max(gt_dp_np))

    amp_norm = (amp_np - amp_np.min()) / (amp_np.max() - amp_np.min() + 1e-8)
    dp_gt_norm = (gt_dp_np - gt_dp_np.min()) / (gt_dp_np.max() - gt_dp_np.min() + 1e-8)
    dp_pr_norm = (pred_dp_np - pred_dp_np.min()) / (pred_dp_np.max() - pred_dp_np.min() + 1e-8)

    # --- Plot upgraded 3×3 grid ---
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)

    titles = [
        'GT Amplitude', 'GT Phase (Phase Offset)', 'GT DP Magnitude',
        'Predicted Amplitude (Fixed)', 'Predicted Phase (Phase Offset)', 'Predicted DP Magnitude',
        'Radial Profile of DP (log)', 'DP Error (Pred - GT)', 'Phase Scatter Plot'
    ]

    colormaps = [
        'Greys', 'twilight_shifted', 'turbo',
        'Greys', 'twilight_shifted', 'turbo',
        None, 'coolwarm', None
    ]

    vmins = [
        0, -np.pi, 0,
        0, -np.pi, 0,
        None, -np.max(np.abs(dp_error)), None
    ]

    vmaxs = [
        1, np.pi, 1,
        1, np.pi, 1,
        None, np.max(np.abs(dp_error)), None
    ]

    images = [
        amp_norm, gt_phase_deramped, dp_gt_norm,
        amp_norm, pred_phase_deramped, dp_pr_norm,
        None, dp_error, None
    ]


    # --- Compute radial profile ---
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
    r_pred = radial_profile(dp_pr_norm)

    # --- Plot all 9 subplots ---
    for idx, ax in enumerate(axes.flat):
        if idx == 6:  # Replace Phase Error with Radial Profile
            ax.plot(r_gt, label="GT", linewidth=2)
            ax.plot(r_pred, label="Pred", linewidth=2)
            ax.set_yscale('log')
            ax.set_title(titles[idx], fontsize=20)
            ax.set_xlabel("Radial Q (px)", fontsize=20)
            ax.set_ylabel("Intensity", fontsize=20)
            ax.tick_params(labelsize=font_size)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=20)
            continue

        elif idx == 8:  # Scatter plot
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
            ax.set_xlim([-np.pi, np.pi])
            ax.set_ylim([-np.pi, np.pi])
            ax.set_xlabel("Ground Truth Phase (rad)", fontsize=20)
            ax.set_ylabel("Predicted Phase (rad)", fontsize=20)
            ax.set_title("Phase Scatter Plot", fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=15, loc="upper left")

            print(f"  Pearson Correlation (R): {pcc:.4f}")
            print(f"  Mean Absolute Error (MAE): {mae:.4f} rad")
            print(f"  Standard Deviation (σ):   {std_dev:.4f} rad\n")

        else:
            im = ax.imshow(images[idx], cmap=colormaps[idx], vmin=vmins[idx], vmax=vmaxs[idx])
            ax.set_title(titles[idx], fontsize=20)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=15)
            cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.01)
            cbar.ax.tick_params(labelsize=15)
            if 'Phase' in titles[idx] or 'Error' in titles[idx]:
                cbar.set_label("Phase (rad)", fontsize=20)
            elif 'Amplitude' in titles[idx]:
                cbar.set_label("Amplitude", fontsize=20)
            elif 'DP' in titles[idx]:
                cbar.set_label("intensity (a.u.)", fontsize=20)

    # --- Save 3x3 output ---
    plt.suptitle(f"Best Run {best_run_idx}: Phase Retrieval Evaluation", fontsize=20)
    os.makedirs("results", exist_ok=True)
    save_figure(fig, f"results/final_3x3_comparison_run_{best_run_idx}", dpi=300, bbox_inches='tight')

    print(f"✅ Final 3x3 comparison with radial profile saved at results/final_3x3_comparison_run_{best_run_idx}.png")

    # Multi-run Metric Histogram Panel
    print("📊 Generating multi-run histogram panel...")
    os.makedirs("results", exist_ok=True)
    from skimage.metrics import structural_similarity as ssim


    def circular_chi_square(pred_np, gt_np, mask):
        diff = (pred_np - gt_np + np.pi) % (2 * np.pi) - np.pi
        return np.mean((diff[mask]) ** 2)


    # --- Start collecting metrics ---
    pcc_phase_all = []
    chisq_phase_all = []
    ssim_phase_all = []

    mask = amp_np > 0
    gt_np = phase_gt.squeeze().cpu().numpy()
    gt_phase_offset = apply_phase_offset(gt_np, mask)

    for run_id in range(NUM_RUNS):
        checkpoint_path = f"checkpoints/single_dp_run_{run_id}.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        with torch.no_grad():
            pred_phase = model(dp_mag, fixed_amplitude)
            pred_dp = compute_fft_dp(pred_phase, fixed_amplitude)

        # Convert to NumPy and offset
        pred_np = pred_phase.squeeze().cpu().numpy()
        pred_offset = apply_phase_offset(pred_np, mask)

        # Phase metrics inside support only
        valid_pred = pred_offset[mask]
        valid_gt = gt_phase_offset[mask]

        # PCC on valid values
        pred_c = valid_pred - np.mean(valid_pred)
        gt_c = valid_gt - np.mean(valid_gt)
        pcc_val = np.sum(pred_c * gt_c) / (np.sqrt(np.sum(pred_c ** 2) * np.sum(gt_c ** 2)) + 1e-8)

        # Chi-square
        chisq_val = circular_chi_square(pred_offset, gt_phase_offset, mask)

        # SSIM (masked) via cropping to bounding box
        y, x = np.where(mask)
        y0, y1 = y.min(), y.max()
        x0, x1 = x.min(), x.max()
        crop_pred = pred_offset[y0:y1 + 1, x0:x1 + 1]
        crop_gt = gt_phase_offset[y0:y1 + 1, x0:x1 + 1]
        crop_mask = mask[y0:y1 + 1, x0:x1 + 1]

        ssim_val = ssim(crop_pred, crop_gt, data_range=2 * np.pi, win_size=11, gaussian_weights=True,
                        use_sample_covariance=False, mask=crop_mask)

        # Append
        pcc_phase_all.append(pcc_val)
        chisq_phase_all.append(chisq_val)
        ssim_phase_all.append(ssim_val)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Row 0
    axes[0, 0].hist(chi_square_list, bins=50, color='#4C72B0', edgecolor='black', alpha=0.85)
    axes[0, 0].set_title(r"$\chi^2$ (DP)", fontsize=20)
    axes[0, 0].set_xlabel(r"$\chi^2$", fontsize=20)
    axes[0, 0].set_ylabel("Count", fontsize=20)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0, 0].tick_params(axis='both', labelsize=15)

    axes[0, 1].hist(pcc_list, bins=50, color='#D55E00', edgecolor='black', alpha=0.85)
    axes[0, 1].set_title("PCC (DP)", fontsize=20)
    axes[0, 1].set_xlabel("PCC", fontsize=20)
    axes[0, 1].set_ylabel("Count", fontsize=20)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0, 1].tick_params(axis='both', labelsize=15)

    axes[0, 2].hist(ssim_phase_all, bins=50, color='#55A868', edgecolor='black', alpha=0.85)
    axes[0, 2].set_title("SSIM (Phase)", fontsize=20)
    axes[0, 2].set_xlabel("SSIM", fontsize=20)
    axes[0, 2].set_ylabel("Count", fontsize=20)
    axes[0, 2].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0, 2].tick_params(axis='both', labelsize=15)

    # Row 1
    axes[1, 0].hist(chisq_phase_all, bins=50, color='#937860', edgecolor='black', alpha=0.85)
    axes[1, 0].set_title(r"$\chi^2$ (Phase)", fontsize=20)
    axes[1, 0].set_xlabel(r"$\chi^2$", fontsize=20)
    axes[1, 0].set_ylabel("Count", fontsize=20)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1, 0].tick_params(axis='both', labelsize=15)

    axes[1, 1].hist(pcc_phase_all, bins=50, color='#C44E52', edgecolor='black', alpha=0.85)
    axes[1, 1].set_title("PCC (Phase)", fontsize=20)
    axes[1, 1].set_xlabel("PCC", fontsize=20)
    axes[1, 1].set_ylabel("Count", fontsize=20)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1, 1].tick_params(axis='both', labelsize=15)

    axes[1, 2].axis('off')  # leave blank

    plt.suptitle("Metric Distributions Across 30 Runs", fontsize=22)
    plt.savefig("results/metrics_histogram_panel_ssim.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: results/metrics_histogram_panel_ssim.png")

    def plot_metric_histogram(data, best_idx, xlabel, title, out_path, color):
        """
        Histogram in blue, no grid, no best-run line.
        X-axis is chi² in [0, 3%].
        """
        from matplotlib.ticker import PercentFormatter
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 5))
        plt.close(fig)

        # Plot raw chi-square (0–0.03) and show as 0–3%
        ax.hist(data, bins=80, color="#5DC468", edgecolor="black", alpha=0.9)

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Frequency", fontsize=13)
        ax.set_title(title, fontsize=14)

        # 0–3% range
        ax.set_xlim(0.0, 0.03)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))  # 0.01 -> 1%
        ax.tick_params(labelsize=20, width=2.0, length=6)
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
        title="Chi-square Distribution (C-ViT runs)",
        out_path=os.path.join(RESULTS_DIR, "chi_square_hist"),
        color="#4C72B0",
    )




