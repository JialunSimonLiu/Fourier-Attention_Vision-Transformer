# ---------------------- 🔹 Imports ----------------------
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
from real_cnnvit_fourier_attention_generalized import *
import torch.nn.init as init

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

# ---------------------- 🔹 Random Seed for Reproducibility ----------------------
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# ---------------------- 🔹 Device Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 🔹 Dataset: Load Phase, Amplitude, DP ----------------------
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

# ---------------------- 🔹 DataLoader ----------------------
def get_single_tif_loader(dp_path="gt_dp_mag_1.tif",
                          amp_path="gt_amp_1.tif",
                          phase_path="gt_phase_1.tif"):
    dataset = SingleTifPhaseDataset(dp_path, amp_path, phase_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader, loader

# ---------------------- 🔹 FFT for DP Reconstruction ----------------------
def compute_fft_dp(phase, fixed_amplitude):
    B, C, H, W = phase.shape
    amplitude_batched = fixed_amplitude.expand(B, 1, H, W)
    density = amplitude_batched * torch.exp(1j * phase)
    dp_mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(density))))
    dp_mag_norm = dp_mag / (dp_mag.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return dp_mag_norm

# ---------------------- 🔹 Save Predicted Phase and DP per Epoch ----------------------
def save_dp_per_epoch(model, test_loader, epoch, fixed_amplitude, save_dir="dp_per_epoch"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    dp_mag, _, _ = next(iter(test_loader))
    dp_mag = dp_mag.to(device)

    with torch.no_grad():
        pred_phase = model(dp_mag, fixed_amplitude)

    pred_dp_mag = compute_fft_dp(pred_phase, fixed_amplitude).cpu().squeeze().numpy()
    pred_phase_np = pred_phase.cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(pred_phase_np, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title(f"Predicted Phase - Epoch {epoch+1}")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(pred_dp_mag, cmap='jet')
    axes[1].set_title(f"Predicted DP - Epoch {epoch+1}")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    # --- SAVE AND FORCE CLOSE ---
    save_path = os.path.join(save_dir, f"epoch_{epoch+1}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ---------------------- 🔹 Custom Hybrid Loss Function ----------------------
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.use_tv = False

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
        # 1. Slow PCC decay → delay Chi² dominance
        beta = (2000.0 / 10.0) * torch.exp(-epoch / 60.0)
        raw_pcc = beta + 1.0

        # 2. PowerChi gradual ramp-up
        if epoch < 10:
            raw_powerchi = torch.tensor(0.0, device=epoch.device)
        elif epoch < 100:
            ramp = torch.sin((epoch - 10) / 90.0 * (np.pi / 2))
            raw_powerchi = torch.tensor(1.0 * ramp.item(), device=epoch.device)
        else:
            raw_powerchi = torch.tensor(1.0, device=epoch.device)

        # 3. Slower Gamma decay for power emphasis
        gamma_start, gamma_end, decay = 2.5, 0.05, 0.01
        if epoch < 10:
            gamma = torch.tensor(0.0, device=epoch.device)
        else:
            gamma = gamma_end + (gamma_start - gamma_end) * torch.exp(-decay * (epoch - 10))

        raw_chisq = torch.tensor(1.0, device=epoch.device)
        return raw_pcc, raw_powerchi, raw_chisq, gamma

    def forward(self, dp_true, y_pred, fixed_amplitude):
        dp_pred = compute_fft_dp(y_pred, fixed_amplitude)
        epoch = torch.tensor(self.epoch, dtype=torch.float32, device=dp_true.device)

        # --- Get raw weights and gamma ---
        raw_pcc, raw_powerchi, raw_chisq, gamma = self.get_raw_weights_and_gamma(epoch)

        # --- Normalize Weights ---
        total = raw_pcc + raw_chisq + raw_powerchi
        w_pcc = raw_pcc / total
        w_chisq = raw_chisq / total
        w_powerchi = raw_powerchi / total

        # --- Compute Losses ---
        loss_pcc = self.pcc_loss(dp_true, dp_pred)
        loss_chisq = self.calculate_chi_square(dp_pred, dp_true)
        loss_powerchi = self.power_chi_square_loss(dp_pred, dp_true, gamma)

        total_loss = (
                w_pcc * loss_pcc +
                w_chisq * loss_chisq +
                w_powerchi * loss_powerchi)

        return total_loss

# ---------------------- 🔹 Training Function ----------------------
def train_phase_model(model, train_loader, test_loader, run_id=0, epochs=1200):
    dp_mag, phase_gt, fixed_amplitude = next(iter(train_loader))
    dp_mag = dp_mag.to(device)
    fixed_amplitude = fixed_amplitude.to(device)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0015, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=1200, eta_min=1e-6)
    loss_fn = HybridLoss()
    scaler = GradScaler()

    loss_history = []
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("loss_plots", exist_ok=True)

    attention_log = []  # Start log for scale weights

    for epoch in range(epochs):
        model.train()
        loss_fn.update_epoch(epoch)
        running_loss = 0.0

        for dp_mag, _, _ in train_loader:
            dp_mag = dp_mag.to(device)

            optimizer.zero_grad()
            with autocast():
                pred_phase = model(dp_mag, fixed_amplitude)
                loss = loss_fn(dp_mag, pred_phase, fixed_amplitude)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        """
        # 🔹 Log multiscale Fourier attention weights
        with torch.no_grad():
            first_block = model.transformer[0].fourier_attn
            if hasattr(first_block, 'weights'):
                weights = F.softmax(first_block.weights, dim=0).cpu().numpy()
                print(f"Epoch {epoch + 1}: Fourier Attention Weights = {weights}")
                attention_log.append(weights.tolist())"""

        #if epoch % 100 == 0:
            #save_dp_per_epoch(model, test_loader, epoch, fixed_amplitude)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    # Save model and loss
    torch.save(model.state_dict(), f"checkpoints/single_dp_run_{run_id}.pth")
    np.save(f"loss_plots/loss_per_epoch_run_{run_id}.npy", np.array(loss_history))
    #np.save(f"loss_plots/attention_weights_run_{run_id}.npy", np.array(attention_log))
    """
    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.title(f"Training Loss Curve (Run {run_id})")
    plt.grid()
    plt.savefig(f"loss_plots/loss_run_{run_id}.png")
    plt.close()
    
    # Plot attention weights
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
        plt.savefig(f"loss_plots/attention_weights_plot_run_{run_id}.png", dpi=300)
        plt.close()"""

    print(f"✅ Finished Training Run {run_id}!")

# ---------------------- 🔹 MAIN: Train 10 Times ----------------------
if __name__ == "__main__":
    for run_id in range(100):
        print(f"\n🔥 Starting Training Run {run_id}/100")
        train_loader, test_loader = get_single_tif_loader()
        model = VisionTransformer(
            img_size=64,
            patch_size=4,  # ↓ reduces tokens from 256 → 64
            in_channels=1,
            embed_dim=128,  # ↓ revert to 128, still enough for 64 tokens
            num_layers=10,  # revert to 8, or keep 10 if convergence needs help
            mlp_dim=256,  # keep this or match embed_dim × 2
            dropout=0.1,
            use_multiscale=True
        )

        model.apply(init_weights)
        train_phase_model(model, train_loader, test_loader, run_id=run_id)
