################################################################################
# Fourier ViT Train file for the synthetic data
# - Jialun Liu, LCN, UCL, 08-10.2025, jialun.liu.17@ucl.ac.uk
################################################################################
# ----------------------  Imports ----------------------
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


# ----------------------  Identity / Ramp Config ----------------------
FORCE_FIXED_AMP = False
USE_FIXED_AMP_UNTIL = 100
PRIOR_MODE = "gaussian"

# ----------------------  Paths from ENV ----------------------
DP_PATH    = os.getenv("DP_PATH", "gt_dp_mag_1.tif")
AMP_PATH   = os.getenv("AMP_PATH", "gt_amp_1.tif")
PHASE_PATH = os.getenv("PHASE_PATH", "gt_phase_1.tif")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOSS_DIR       = os.path.join(OUTPUT_DIR, "loss_plots")
DPEPOCH_DIR    = os.path.join(OUTPUT_DIR, "dp_per_epoch")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)
os.makedirs(DPEPOCH_DIR, exist_ok=True)

print(f"[paths] DP_PATH={DP_PATH}")
print(f"[paths] AMP_PATH={AMP_PATH}")
print(f"[paths] PHASE_PATH={PHASE_PATH}")
print(f"[paths] OUTPUT_DIR={OUTPUT_DIR}")
for label, p in [("DP_PATH", DP_PATH), ("AMP_PATH", AMP_PATH), ("PHASE_PATH", PHASE_PATH)]:
    if not os.path.exists(p.split(",")[0]):  # allow multi-DP comma list
        raise FileNotFoundError(f"{label} not found: {p}")

# ----------------------  Init Weights ----------------------
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.out_channels == 2:
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

# ----------------------   Device Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------  Dataset ----------------------
class MultiTifPhaseDataset(Dataset):
    def __init__(self, dp_paths, amp_path, phase_path):
        if isinstance(dp_paths, str):
            dp_paths = [p.strip() for p in dp_paths.split(",")]
        self.dp_paths = dp_paths

        amp = tiff.imread(amp_path).astype(np.float32)
        amp = amp / np.max(amp)
        self.amp = torch.tensor(amp).unsqueeze(0)

        phase = tiff.imread(phase_path).astype(np.float32)
        phase = (phase / 65535.0) * (2 * np.pi) - np.pi
        self.phase = torch.tensor(phase).unsqueeze(0)

    def __len__(self):
        return len(self.dp_paths)

    def __getitem__(self, idx):
        dp = tiff.imread(self.dp_paths[idx]).astype(np.float32)
        dp = dp / np.max(dp)
        dp = torch.tensor(dp).unsqueeze(0)
        return dp, self.phase, self.amp


def get_loader(dp_paths=DP_PATH, amp_path=AMP_PATH, phase_path=PHASE_PATH):
    dataset = MultiTifPhaseDataset(dp_paths, amp_path, phase_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader, loader


# ----------------------  Prior (flat / blurred / gaussian) ----------------------
def make_gaussian_prior(support, jitter=False):
    bin_sup = (support > 0).float()
    B, C, H, W = bin_sup.shape
    y, x = torch.meshgrid(torch.arange(H, device=bin_sup.device),
                          torch.arange(W, device=bin_sup.device), indexing='ij')
    idx = (bin_sup[0, 0] > 0)
    cy, cx = y[idx].float().mean(), x[idx].float().mean()
    sigma = (0.35 + (0.1 * torch.rand(1, device=bin_sup.device) if jitter else 0.0)) * float(max(H, W))
    g = torch.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sigma**2))
    g = g / (g.max() + 1e-8)
    return g.unsqueeze(0).unsqueeze(0) * bin_sup


def make_blurred_prior(support, k=5, iters=3):
    bin_sup = (support > 0).float()
    prior = bin_sup.clone()
    for _ in range(iters):
        prior = F.avg_pool2d(prior, k, stride=1, padding=k // 2)
    prior = prior / (prior.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return prior * bin_sup


def make_prior(support):
    bin_sup = (support > 0).float()
    if PRIOR_MODE == "flat":
        return bin_sup
    elif PRIOR_MODE == "blurred":
        return make_blurred_prior(bin_sup)
    elif PRIOR_MODE == "gaussian":
        return make_gaussian_prior(bin_sup)
    else:
        raise ValueError(f"Unknown PRIOR_MODE: {PRIOR_MODE}")

# ----------------------  FFT for DP Reconstruction ----------------------
def compute_fft_dp(phase, amplitude):
    density = amplitude * torch.exp(1j * phase)
    dp_mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(density), norm="ortho")))
    dp_mag_norm = dp_mag / (dp_mag.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return dp_mag_norm

# ----------------------  Custom Hybrid Loss Function ----------------------
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.use_tv = True

    def update_epoch(self, epoch):
        self.epoch = epoch

    def pcc_loss(self, target_dp, predicted_dp):
        x = predicted_dp - predicted_dp.mean(dim=(-1, -2), keepdim=True)
        y = target_dp - target_dp.mean(dim=(-1, -2), keepdim=True)
        numerator = (x * y).sum(dim=(-1, -2))
        denominator = torch.sqrt((x ** 2).sum(dim=(-1, -2)) * (y ** 2).sum(dim=(-1, -2)) + 1e-8)
        return (1 - numerator / denominator).mean()

    def calculate_chi_square(self, predicted_dp, target_dp):
        predicted_norm = predicted_dp / (predicted_dp.amax(dim=(-1, -2), keepdim=True) + 1e-8)
        target_norm = target_dp / (target_dp.amax(dim=(-1, -2), keepdim=True) + 1e-8)
        chi_square = torch.sum((predicted_norm - target_norm) ** 2, dim=(-1, -2)) / (
                torch.sqrt(
                    torch.sum(predicted_norm ** 2, dim=(-1, -2)) * torch.sum(target_norm ** 2, dim=(-1, -2))) + 1e-8
        )
        return chi_square.mean()

    def weighted_chi_square_loss(self, dp_pred, dp_true, weight_map):
        return ((weight_map * (dp_pred - dp_true) ** 2) / (dp_true + 1e-8)).mean()

    def power_chi_square_loss(self, dp_pred, dp_true, gamma):
        dp_pred_powered = dp_pred ** gamma
        dp_true_powered = dp_true ** gamma

        dp_pred_norm = dp_pred_powered / (dp_pred_powered.amax(dim=(-1, -2), keepdim=True) + 1e-8)
        dp_true_norm = dp_true_powered / (dp_true_powered.amax(dim=(-1, -2), keepdim=True) + 1e-8)

        num = torch.sum((dp_pred_norm - dp_true_norm) ** 2, dim=(-1, -2))
        denom = torch.sqrt(torch.sum(dp_pred_norm ** 2, dim=(-1, -2)) * torch.sum(dp_true_norm ** 2, dim=(-1, -2)) + 1e-8)
        return (num / denom).mean()

    def total_variation_loss(self, img):
        tv_x = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        tv_y = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        return (tv_x.mean() + tv_y.mean())

    def get_raw_weights_and_gamma(self, epoch):
        # 1) PCC Exponential Decay — slower so it remains useful longer
        beta = (2000.0 / 10.0) * torch.exp(-epoch / 15.0)
        raw_pcc = beta + 1.0

        # 2) Power-Chi Ramp — quicker ramp so χ² helps earlier
        power_peak = 0.5
        if epoch < 10:
            raw_powerchi = torch.tensor(0.0, device=epoch.device)
        elif epoch < 30:
            ramp = torch.sin((epoch - 10) / 20.0 * (np.pi / 2))
            raw_powerchi = torch.tensor(power_peak * ramp.item(), device=epoch.device)
        else:
            raw_powerchi = torch.tensor(power_peak, device=epoch.device)

        # 3) Gamma Decay — accelerate so γ<1 by ~epoch 100
        gamma_start, gamma_end, decay = 3.0, 0.01, 0.03
        if epoch < 10:
            gamma = torch.tensor(0.0, device=epoch.device)
        else:
            gamma = gamma_end + (gamma_start - gamma_end) * torch.exp(-decay * (epoch - 10))

        # raw_chisq stays at 1.0
        raw_chisq = torch.tensor(1.0, device=epoch.device)

        return raw_pcc, raw_powerchi, raw_chisq, gamma

    def forward(self, dp_true, amp_pred, pha_pred, fixed_support):
        # --- Compute predicted DP ---
        dp_pred = compute_fft_dp(pha_pred, amp_pred)
        epoch = torch.tensor(self.epoch, dtype=torch.float32, device=dp_true.device)

        # --- Your existing hybrid loss components ---
        loss_pcc = self.pcc_loss(dp_true, dp_pred)
        loss_chisq = self.calculate_chi_square(dp_pred, dp_true)
        loss_pchi = self.power_chi_square_loss(dp_pred, dp_true, self.get_raw_weights_and_gamma(epoch)[3])

        raw_pcc, raw_powerchi, raw_chisq, _ = self.get_raw_weights_and_gamma(epoch)
        total = raw_pcc + raw_powerchi + raw_chisq
        w_pcc, w_pchi, w_chi = raw_pcc / total, raw_powerchi / total, raw_chisq / total

        # ==========================================================
        # 🔹 New: amplitude regularizers
        # ==========================================================
        amp_in = amp_pred * fixed_support

        # Anti-hotspot penalty (most effective) ---
        μ = amp_in.mean()
        σ = amp_in.std(unbiased=False) + 1e-6
        σ = torch.clamp(σ, min=0.02)  # small floor
        excess = F.relu(amp_in - (μ + 3.0 * σ))
        L_hot = (excess ** 2).mean()  # tiny weight ~0.005

        # Total-variation smoothing (smaller after 500) ---
        tv_w = 0.2 if epoch < 500 else 0.05
        tv_loss = self.total_variation_loss(amp_pred)
        # ==========================================================

        # --- Combine everything (keep your structure) ---
        total_loss = (
                w_pcc * loss_pcc +
                w_chi * loss_chisq +
                w_pchi * loss_pchi +
                tv_w * tv_loss
        )

        return total_loss


# ---------------------- 🔹 Training ----------------------
def train_phase_model(model, train_loader, test_loader, run_id=0, epochs=1000):
    dp_mag, phase_gt, fixed_support = next(iter(train_loader))
    dp_mag, fixed_support = dp_mag.to(device), fixed_support.to(device)
    prior_amp = make_prior(fixed_support)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    loss_fn = HybridLoss()
    scaler = GradScaler()
    loss_history = []

    for epoch in range(epochs):
        model.train()
        loss_fn.update_epoch(epoch)
        running_loss = 0.0
        for dp_mag, _, _ in train_loader:
            dp_mag = dp_mag.to(device)
            optimizer.zero_grad()
            with autocast():
                raw_amp, pred_phase = model(dp_mag, fixed_support, prior_amp=prior_amp)
                pred_amp = prior_amp.clone() if (FORCE_FIXED_AMP or epoch <= USE_FIXED_AMP_UNTIL) else raw_amp
                loss = loss_fn(dp_mag, pred_amp, pred_phase, fixed_support)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"single_dp_run_{run_id}.pth"))
    np.save(os.path.join(LOSS_DIR, f"loss_per_epoch_run_{run_id}.npy"), np.array(loss_history))
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.title(f"Training Loss Curve (Run {run_id})")
    plt.grid()
    plt.savefig(os.path.join(LOSS_DIR, f"loss_run_{run_id}.png"))
    plt.close()
    print(f"✅ Finished Training Run {run_id}!")


# ---------------------- 🔹 MAIN ----------------------
if __name__ == "__main__":
    for run_id in range(100):
        print(f"\n Starting Training Run {run_id}/100")
        torch.manual_seed(run_id)
        np.random.seed(run_id)
        random.seed(run_id)
        train_loader, test_loader = get_loader()
        model = VisionTransformer(
            img_size=64,
            patch_size=4,
            in_channels=1,
            embed_dim=128,
            num_layers=10,
            mlp_dim=256,
            dropout=0.1,
            use_multiscale=True
        )
        model.apply(init_weights)
        train_phase_model(model, train_loader, test_loader, run_id=run_id)
