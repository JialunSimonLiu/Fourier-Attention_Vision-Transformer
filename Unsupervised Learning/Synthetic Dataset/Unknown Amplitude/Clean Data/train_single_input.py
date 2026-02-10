################################################################################
# Fourier ViT Train file for the synthetic data
# - Jialun Liu, LCN, UCL, 08-10.2025, jialun.liu.17@ucl.ac.uk
################################################################################
# ---------------------- Imports ----------------------
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import tifffile as tiff
import matplotlib
matplotlib.use('Agg')  # non-interactive for server running
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from real_cnnvit_fourier_attention_amp_generalized import *
import torch.nn.init as init
import torch.nn.functional as F
import math

# --- Per-dataset output root (set by run_multi_experiment_dp.py) ---
BASE_OUT = os.getenv("OUTPUT_DIR", ".")
CHECKPOINT_DIR = os.path.join(BASE_OUT, "checkpoints")
LOSS_DIR = os.path.join(BASE_OUT, "loss_plots")
PER_EPOCH_DIR = os.path.join(BASE_OUT, "dp_per_epoch")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)
os.makedirs(PER_EPOCH_DIR, exist_ok=True)

# ---------------------- Identity / Ramp Config ----------------------
FORCE_FIXED_AMP = False    # no longer used for blending but kept for compatibility
PRIOR_MODE = "gaussian"    # "flat" | "blurred" | "gaussian"

# ---------------------- Amplitude prior blend schedule ----------------------
AMP_LOCK_EPOCH = 10#50
AMP_PRIOR_DECAY_END = 50#200
ALPHA_START = 1.0
ALPHA_END   = 0.0

def get_alpha(epoch: int) -> float:
    if epoch < AMP_LOCK_EPOCH:
        return ALPHA_START
    if epoch >= AMP_PRIOR_DECAY_END:
        return ALPHA_END
    t = (epoch - AMP_LOCK_EPOCH) / (AMP_PRIOR_DECAY_END - AMP_LOCK_EPOCH)
    # cosine from ALPHA_START to ALPHA_END
    return ALPHA_END + (ALPHA_START - ALPHA_END) * 0.5 * (1 + math.cos(math.pi * t))


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.out_channels == 2:
            # Small random init to enable learning signal
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        else:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ----------------------  Device Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------  Dataset: Load Phase, Amplitude, DP ----------------------
class SingleTifPhaseDataset(Dataset):
    def __init__(self, dp_path, amp_path, phase_path):
        self.dp = tiff.imread(dp_path).astype(np.float32)
        self.dp = self.dp / np.max(self.dp)
        self.dp = torch.tensor(self.dp).unsqueeze(0)

        self.amp = tiff.imread(amp_path).astype(np.float32)
        self.amp = self.amp / np.max(self.amp)
        self.amp = torch.tensor(self.amp).unsqueeze(0)

        if phase_path is not None and os.path.exists(phase_path):
            phase = tiff.imread(phase_path).astype(np.float32)
            phase = (phase / 65535.0) * (2 * np.pi) - np.pi
        else:
            phase = np.zeros_like(self.dp.squeeze(0).numpy(), dtype=np.float32)
        self.phase = torch.tensor(phase).unsqueeze(0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.dp, self.phase, self.amp

# ----------------------  DataLoader ----------------------
def get_single_tif_loader(dp_path="gt_dp_mag_1.tif",
                          amp_path="gt_amp_1.tif",
                          phase_path="gt_phase_1.tif"):
    # pick up env vars from run_multi_experiment_dp.py
    dp_path_env   = os.getenv("DP_PATH", dp_path)
    amp_path_env  = os.getenv("AMP_PATH", amp_path)
    phase_path_env = os.getenv("PHASE_PATH", phase_path)

    # If PHASE_PATH is unset or file missing, dataset will create zeros
    if not phase_path_env or not os.path.exists(phase_path_env):
        phase_path_env = None

    dataset = SingleTifPhaseDataset(dp_path_env, amp_path_env, phase_path_env)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader, loader

# ----------------------  prior (flat/blurred/gaussian built from a binary mask) ----------------------
def make_gaussian_prior(support, jitter=False):
    # use a flat (binary) support: 1 inside, 0 outside
    bin_sup = (support > 0).float()
    B, C, H, W = bin_sup.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=bin_sup.device),
        torch.arange(W, device=bin_sup.device),
        indexing='ij'
    )

    # centroid within the support region
    idx = (bin_sup[0, 0] > 0)
    cy = y[idx].float().mean()
    cx = x[idx].float().mean()

    # Gaussian width: moderate, with small jitter for run diversity
    sigma = (0.35 + (0.1 * torch.rand(1, device=bin_sup.device) if jitter else 0.0)) * float(max(H, W))

    g = torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))
    g = g / (g.max() + 1e-8)

    # mask outside the support
    return g.unsqueeze(0).unsqueeze(0) * bin_sup


def make_blurred_prior(support, k=5, iters=3):
    # start from flat (binary) support, then blur → soft edge prior
    bin_sup = (support > 0).float()
    prior = bin_sup.clone()
    for _ in range(iters):
        prior = F.avg_pool2d(prior, k, stride=1, padding=k // 2)
    # normalise to [0,1] and keep inside the (flat) support
    prior = prior / (prior.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return prior * bin_sup


def make_prior(support):
    # always reduce the provided amplitude map to a flat (binary) support first
    bin_sup = (support > 0).float()
    if PRIOR_MODE == "flat":
        return bin_sup
    elif PRIOR_MODE == "blurred":
        # blurred is applied on the flat version
        return make_blurred_prior(bin_sup)
    elif PRIOR_MODE == "gaussian":
        # gaussian is applied on the flat version
        return make_gaussian_prior(bin_sup)
    else:
        raise ValueError(f"Unknown PRIOR_MODE: {PRIOR_MODE}")

# ---------------------- FFT for DP Reconstruction ----------------------
def compute_fft_dp(phase, amplitude):
    density = amplitude * torch.exp(1j * phase)
    dp_mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(density), norm="ortho")))
    dp_mag_norm = dp_mag / (dp_mag.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return dp_mag_norm

# ----------------------  Save Predicted Phase and DP per Epoch ----------------------
# --- Add a dedicated folder for epoch-wise TIFF outputs ---
PER_EPOCH_TIF_DIR = os.path.join(PER_EPOCH_DIR, "tif")
os.makedirs(PER_EPOCH_TIF_DIR, exist_ok=True)

def phase_to_uint16(phase_rad: np.ndarray) -> np.ndarray:
    """
    Map phase in [-pi, pi] to uint16 [0, 65535]
    """
    phase_wrapped = (phase_rad + np.pi) % (2 * np.pi) - np.pi  # ensure in [-pi, pi]
    u16 = np.round((phase_wrapped + np.pi) / (2 * np.pi) * 65535.0).astype(np.uint16)
    return u16

def float01_to_uint16(x: np.ndarray) -> np.ndarray:
    """
    Clip float to [0,1] and save as uint16 [0,65535]
    """
    x = np.clip(x, 0.0, 1.0)
    return np.round(x * 65535.0).astype(np.uint16)

def save_dp_per_epoch(model, test_loader, epoch, fixed_support, save_dir=PER_EPOCH_DIR):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    dp_mag, _, _ = next(iter(test_loader))
    dp_mag = dp_mag.to(device)
    fixed_support = fixed_support.to(device)
    prior_amp = make_prior(fixed_support).to(device)

    with torch.no_grad():
        raw_amp, pred_phase = model(dp_mag, fixed_support, prior_amp=prior_amp)

        # Alpha blending between prior and learned amplitude
        alpha = get_alpha(epoch)
        pred_amp = alpha * prior_amp + (1.0 - alpha) * raw_amp

    pred_dp_mag = compute_fft_dp(pred_phase, pred_amp).cpu().squeeze().numpy()
    pred_phase_np = pred_phase.cpu().squeeze().numpy()
    pred_amp_np = pred_amp.cpu().squeeze().numpy()

    e = epoch + 1
    amp_u16 = float01_to_uint16(pred_amp_np)
    dp_u16 = float01_to_uint16(pred_dp_mag)
    phase_u16 = phase_to_uint16(pred_phase_np)

    tiff.imwrite(os.path.join(PER_EPOCH_TIF_DIR, f"amp_epoch_{e:04d}.tif"), amp_u16)
    tiff.imwrite(os.path.join(PER_EPOCH_TIF_DIR, f"dp_epoch_{e:04d}.tif"), dp_u16)
    tiff.imwrite(os.path.join(PER_EPOCH_TIF_DIR, f"phase_epoch_{e:04d}.tif"), phase_u16)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Amplitude
    im1 = axes[0].imshow(pred_amp_np, cmap='Greys', vmin=0, vmax=1)
    axes[0].set_title(f"Predicted Amplitude - Epoch {epoch + 1}")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # Phase
    im2 = axes[1].imshow(pred_phase_np, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"Predicted Phase - Epoch {epoch + 1}")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    # Diffraction Pattern
    im3 = axes[2].imshow(pred_dp_mag, cmap='turbo')
    axes[2].set_title(f"Predicted DP - Epoch {epoch + 1}")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()

    save_path = os.path.abspath(os.path.join(save_dir, f"epoch_{epoch + 1}.png"))
    try:
        plt.savefig(save_path, dpi=300)
    except OSError:
        fallback = save_path.replace(".png", f"_fallback_{np.random.randint(1000)}.png")
        print(f"Could not save {save_path}, fallback to {fallback}")
        plt.savefig(fallback, dpi=300)
    plt.close(fig)

# ---------------------- Custom Hybrid Loss Function ----------------------
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.use_tv = True

    def update_epoch(self, epoch):
        self.epoch = epoch

    def pcc_loss(self, target_dp, predicted_dp):
        eps = 1e-8
        # subtract mean
        x = predicted_dp - predicted_dp.mean(dim=(-1, -2), keepdim=True)
        y = target_dp - target_dp.mean(dim=(-1, -2), keepdim=True)
        # correlation numerator
        numerator = (x * y).sum(dim=(-1, -2))
        # denominator (variances)
        denom = torch.sqrt((x ** 2).sum(dim=(-1, -2)) *
                           (y ** 2).sum(dim=(-1, -2)) + eps)
        pcc = numerator / denom
        return (1 - pcc).mean()

    def calculate_chi_square(self, predicted_dp, target_dp):

        eps = 1e-8
        # Convert to magnitude just in case the tensor is complex
        A = torch.abs(target_dp)
        B = torch.abs(predicted_dp)

        # RMS normalisation
        A_rms = torch.sqrt(torch.mean(A ** 2, dim=(-1, -2), keepdim=True) + eps)
        B_rms = torch.sqrt(torch.mean(B ** 2, dim=(-1, -2), keepdim=True) + eps)

        A_norm = A / A_rms
        B_norm = B / B_rms

        # Chi-square: sum((A-B)^2) / sum(A^2)
        numerator = torch.sum((A_norm - B_norm) ** 2, dim=(-1, -2))
        denominator = torch.sum(A_norm ** 2, dim=(-1, -2)) + eps

        return (numerator / denominator).mean()

    def power_chi_square_loss(self, dp_pred, dp_true, gamma):
        eps = 1e-8
        # magnitudes
        A = torch.abs(dp_true)
        B = torch.abs(dp_pred)

        # RMS-normalise before power
        A_rms = torch.sqrt(torch.mean(A ** 2, dim=(-1, -2), keepdim=True) + eps)
        B_rms = torch.sqrt(torch.mean(B ** 2, dim=(-1, -2), keepdim=True) + eps)

        A_norm = A / A_rms
        B_norm = B / B_rms

        # apply power AFTER normalisation
        A_pow = A_norm ** gamma
        B_pow = B_norm ** gamma

        # chi-square: sum((A-B)^2) / sum(A^2)
        numerator = torch.sum((A_pow - B_pow) ** 2, dim=(-1, -2))
        denominator = torch.sum(A_pow ** 2, dim=(-1, -2)) + eps

        return (numerator / denominator).mean()

    def total_variation_loss(self, amp, support_mask=None, eps=1e-8):
        # Respect support (optional)
        if support_mask is not None:
            amp = amp * support_mask

        # Forward differences
        tv_x = torch.abs(amp[:, :, 1:, :] - amp[:, :, :-1, :])
        tv_y = torch.abs(amp[:, :, :, 1:] - amp[:, :, :, :-1])

        # Edge-preserving anisotropic TV
        return (tv_x.mean() + tv_y.mean())

    def get_raw_weights_and_gamma(self, epoch: torch.Tensor):
        # make sure it's float tensor
        epoch = epoch.float()

        # 1) PCC Exponential Decay — slower so it remains useful longer
        beta = 200 * torch.exp(-epoch / 15.0)
        raw_pcc = beta + 1.0

        # 2) Power-Chi
        power_peak = 0.5
        # smooth ramp using sin, same style as before
        phase = (epoch.clamp(0.0, 20.0) / 20.0) * (torch.pi / 2.0)
        ramp = torch.sin(phase).clamp(0.0, 1.0)
        raw_powerchi = power_peak * ramp  # 0 at epoch 0, ≈0.5 by epoch ~20

        # 3) Gamma Decay — from 2.0 at epoch 0, decays so γ < 1 by ~epoch 100
        gamma_start, gamma_end, decay = 2.0, 0.5, 0.005
        gamma = gamma_end + (gamma_start - gamma_end) * torch.exp(-decay * epoch)

        # 4) Standard chi-square: constant raw weight 1.0
        raw_chisq = torch.ones_like(epoch)

        # 5) Late-phase rebalance after epoch 200 (unchanged from your version)
        if epoch.item() >= 200:
            scale_pcc = 0.6  # halve PCC 0.4
            scale_chi = 1.1  # slightly favour χ² 1.2
            scale_pchi = 0.6  # keep high-q polishing 1.0

            raw_pcc = raw_pcc * scale_pcc
            raw_chisq = raw_chisq * scale_chi
            raw_powerchi = raw_powerchi * scale_pchi

        return raw_pcc, raw_powerchi, raw_chisq, gamma

    def forward(self, dp_true, amp_pred, pha_pred, fixed_support):

        # keep as float32 tensor on correct device
        epoch = torch.tensor(self.epoch, dtype=torch.float32, device=dp_true.device)

        # Compute predicted DP
        dp_pred = compute_fft_dp(pha_pred, amp_pred)

        # Loss components
        loss_pcc = self.pcc_loss(dp_true, dp_pred)
        loss_chisq = self.calculate_chi_square(dp_pred, dp_true)

        raw_pcc, raw_powerchi, raw_chisq, gamma = self.get_raw_weights_and_gamma(epoch)

        loss_pchi = self.power_chi_square_loss(dp_pred, dp_true, gamma)

        total = raw_pcc + raw_powerchi + raw_chisq
        w_pcc = raw_pcc / total
        w_pchi = raw_powerchi / total
        w_chi = raw_chisq / total

        # Total-variation smoothing (smaller after 500)
        tv_w = 0.01 if self.epoch < 200 else 0.0 # experiment
        #tv_w = 0.2 if self.epoch < 500 else 0.05 # simulation
        #tv_w = 0.0
        tv_loss = self.total_variation_loss(amp_pred, fixed_support)

        # --- Combine everything (exactly your structure) ---
        total_loss = (
            w_pcc * loss_pcc +
            w_chi * loss_chisq +
            w_pchi * loss_pchi +
            tv_w * tv_loss
        )

        return total_loss

# ---------------------- Training Function ----------------------
def train_phase_model(model, train_loader, test_loader, run_id=0, epochs=1000):
    # --- Get the single sample once ---
    dp_mag, phase_gt, fixed_support = next(iter(train_loader))
    dp_mag = dp_mag.to(device)
    fixed_support = fixed_support.to(device)

    # neutral amplitude prior once
    prior_amp = make_prior(fixed_support).to(device)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = HybridLoss()
    scaler = GradScaler()

    loss_history = []

    for epoch in range(epochs):
        model.train()
        loss_fn.update_epoch(epoch)

        optimizer.zero_grad()
        with autocast():

            raw_amp, pred_phase = model(dp_mag, fixed_support, prior_amp=prior_amp)

            # Alpha blending between prior and learned amplitude
            alpha = get_alpha(epoch)
            pred_amp = alpha * prior_amp + (1.0 - alpha) * raw_amp

            loss = loss_fn(dp_mag, pred_amp, pred_phase, fixed_support)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        avg_loss = loss.item()
        loss_history.append(avg_loss)

        """
        # Log multiscale Fourier attention weights (kept commented as in your version)
        with torch.no_grad():
            first_block = model.transformer[0].fourier_attn
            if hasattr(first_block, 'weights'):
                weights = F.softmax(first_block.weights, dim=0).cpu().numpy()
                print(f"Epoch {epoch + 1}: Fourier Attention Weights = {weights}")
                attention_log.append(weights.tolist())
        """

        #if (epoch < 250 and epoch % 20 == 0) or (epoch >= 250 and epoch % 100 == 0):
        save_dp_per_epoch(model, test_loader, epoch, fixed_support)

        if (epoch + 1) % 200 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Run {run_id:03d} | Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

    # Save model and loss
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"single_dp_run_{run_id}.pth"))
    np.save(os.path.join(LOSS_DIR, f"loss_per_epoch_run_{run_id}.npy"), np.array(loss_history))
    # np.save(os.path.join(LOSS_DIR, f"attention_weights_run_{run_id}.npy"), np.array(attention_log))
    """
    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.title(f"Training Loss Curve (Run {run_id})")
    plt.grid()
    plt.savefig(os.path.join(LOSS_DIR, f"loss_run_{run_id}.png"))
    plt.close()

    # Plot attention weights (if you re-enable logging above)
    if attention_log:
        attention_log = np.array(attention_log)
        plt.figure(figsize=(8, 5))
        for i, label in enumerate(["Full Scale (1x)", "Half Scale (1/2x)", "Quarter Scale (1/4x)"]):
            plt.plot(attention_log[:, i], label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Attention Weight")
        plt.title(f"Fourier Attention Weights (Run {run_id})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(LOSS_DIR, f"attention_weights_plot_run_{run_id}.png"), dpi=300)
        plt.close()
    """

    print(f"Finished Training Run {run_id}!")

# ---------------------- MAIN: Train 5 Times ----------------------
if __name__ == "__main__":
    train_loader, test_loader = get_single_tif_loader()
    NUM_RUNS = 100
    for run_id in range(NUM_RUNS):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"single_dp_run_{run_id}.pth")
        if os.path.exists(ckpt_path):
            print(f" Run {run_id:03d} already exists, skipping.")
            continue

        print(f"\n Starting Training Run {run_id}/{NUM_RUNS-1}")
        torch.manual_seed(run_id)
        np.random.seed(run_id)
        random.seed(run_id)

        model = VisionTransformer(
            img_size=64,
            patch_size=4,
            in_channels=1,
            embed_dim=128,
            num_layers=10,
            mlp_dim=256,
            dropout=0.0,
            use_multiscale=True
        )
        model.apply(init_weights)
        train_phase_model(model, train_loader, test_loader, run_id=run_id)


