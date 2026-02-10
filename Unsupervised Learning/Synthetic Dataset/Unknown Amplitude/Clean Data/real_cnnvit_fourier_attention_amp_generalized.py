################################################################################
# The core of Fourier ViT
# - Jialun Liu, LCN, UCL, 03-10.2025, jialun.liu.17@ucl.ac.uk
################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- Real-Space Mask (Support Constraint) ----------------------
def apply_support_mask(tensor, support_amplitude):
    # Apply binary mask where amplitude is nonzero (domain support region)
    mask = (support_amplitude > 0.0).float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)              # (B, 1, H, W)
    elif mask.ndim != 4:
        raise ValueError("Invalid support_amplitude shape")
    return tensor * mask

# ---------------------- CNN Feature Extractor ----------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # 64×64
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)           # 64×64
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, embed_dim, kernel_size=1)               # 64×64
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))  # low-level edges
        x = self.relu2(self.conv2(x))  # mid-level context
        x = self.relu3(self.conv3(x))  # project to embed_dim
        return x

# ---------------------- Patch Embedding with Skip Connection ----------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=128):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(in_channels, embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat = self.feature_extractor(x)           # CNN skip connection
        x_proj = self.proj(feat)                   # Patchify
        B, C, H, W = x_proj.shape
        tokens = x_proj.flatten(2).transpose(1, 2) # (B, N, E)
        return self.norm(tokens), feat, (H, W)

# ---------------------- CNN Decoder with FFT Feature Fusion ----------------------
class CNNDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Increase decoder channels to handle 64×64 resolution
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.deconv1 = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)  # from 128 to 256

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.deconv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)        # from 64 to 128

        self.fuse = nn.Sequential(
            nn.Conv2d(128 + embed_dim * 2, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.output_amp = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.output_phase = nn.Conv2d(64, 2, kernel_size=3, padding=1) # Output cos/sin phase
        #nn.init.constant_(self.output_amp.bias, 0.2) # Initial amplitude approx. equals to 0.5

    def forward(self, tokens, skip, fixed_support, prior_amp=None):
        B, N, E = tokens.shape
        side = int(N**0.5)
        x = tokens.view(B, E, side, side)

        x = self.upsample1(F.relu(self.deconv1(x)))
        x = self.upsample2(F.relu(self.deconv2(x)))
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        # Fourier magnitude of skip (real-space info injection)
        #skip_fft_mag = torch.abs(torch.fft.fft2(skip.to(torch.float32)))
        #x = torch.cat([x, skip, skip_fft_mag], dim=1)
        # New: stable FFT features, same shapes
        skip_c = skip.to(torch.float32)
        skip_fft = torch.fft.fftshift(torch.fft.fft2(skip_c, norm="ortho"))
        skip_fft_mag = torch.log1p(skip_fft.abs())
        skip_fft_mag = (skip_fft_mag - skip_fft_mag.mean(dim=(2, 3), keepdim=True)) / \
                       (skip_fft_mag.std(dim=(2, 3), keepdim=True) + 1e-6)
        x = torch.cat([x, skip_c, skip_fft_mag], dim=1)

        x = self.fuse(x)
        x = self.refine(x)


        mask = (fixed_support > 0).float()
        eps = 1e-8

        amp = F.softplus(self.output_amp(x))
        #amp = F.sigmoid(self.output_amp(x))
        amp = apply_support_mask(amp, mask)

        #amp = amp / (amp.amax(dim=(-1,-2), keepdim=True) + 1e-8)

        # --- use prior_amp (soft support) if provided; otherwise flat mask ---
        if prior_amp is not None:
            prior = prior_amp.to(amp.device)
            prior = apply_support_mask(prior, mask)
        else:
            prior = mask

        E_cur = torch.sqrt(((amp * mask) ** 2).sum(dim=(-1, -2), keepdim=True) + eps)
        E_tgt = torch.sqrt(((prior * mask) ** 2).sum(dim=(-1, -2), keepdim=True) + eps)

        scale = (E_tgt / E_cur)
        scale = scale.clamp(0.8, 1.2)  # soft bounds turn on for experiment
        amp = amp * scale

        phase_twochan = self.output_phase(x)  # (B,2,H,W)
        return amp, phase_twochan

# ---------------------- Flexible Spectral Attention Mixer ----------------------
class FourierAttentionWithMask(nn.Module):
    def __init__(self, dim, height, width):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width
        self.fourier_weight = nn.Parameter(torch.empty(1, dim, height, width))
        nn.init.normal_(self.fourier_weight, mean=1.0, std=0.05)
        self.freq_mask = nn.Parameter(torch.ones(1, 1, height, width))

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm="ortho")
        modulated = x_fft * self.fourier_weight * self.freq_mask
        x_ifft = torch.fft.ifft2(modulated, norm="ortho").real
        return x_ifft  # (B, C, H, W)

class FlexibleSpectralAttentionMixer(nn.Module):
    def __init__(self, dim, height, width, num_scales=3):
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
        self.scales = [1, 2, 4][:num_scales]
        self.weights = nn.Parameter(torch.zeros(len(self.scales)))
        self.fourier_modules = nn.ModuleList([
            FourierAttentionWithMask(dim, height // s, width // s) for s in self.scales
        ])

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.height, self.width
        x_spatial = x.transpose(1, 2).view(B, C, H, W)
        outputs = []
        for i, scale in enumerate(self.scales):
            x_scaled = F.avg_pool2d(x_spatial, scale) if scale > 1 else x_spatial
            out = self.fourier_modules[i](x_scaled)
            if scale > 1:
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            outputs.append(out.view(B, C, -1).transpose(1, 2))  # (B, N, C)

        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * o for w, o in zip(weights, outputs))
        return fused

class FourierTokenWrapper(nn.Module):
    def __init__(self, module, height, width):
        super().__init__()
        self.module = module
        self.height = height
        self.width = width

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, self.height, self.width)
        out_2d = self.module(x_2d)
        return out_2d.view(B, C, -1).transpose(1, 2)  # return [B, N, C]

# ---------------------- Fourier Transformer Block ----------------------
class FourierTransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, height=8, width=8, mlp_dim=128, dropout=0.0, use_multiscale=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        if use_multiscale:
            self.fourier_attn = FlexibleSpectralAttentionMixer(embed_dim, height, width, num_scales=3)
        else:
            fa_module = FourierAttentionWithMask(embed_dim, height, width)
            self.fourier_attn = FourierTokenWrapper(fa_module, height, width)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.fourier_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------- Vision Transformer ----------------------
class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=128,
                 num_layers=8, mlp_dim=256, dropout=0.0, use_multiscale=True):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.token_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        H, W = img_size // patch_size, img_size // patch_size
        self.grid_size = (H, W)
        num_patches = H * W
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.transformer = nn.ModuleList([
            FourierTransformerBlock(embed_dim, height=H, width=W, mlp_dim=mlp_dim, dropout=dropout, use_multiscale=use_multiscale)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = CNNDecoder(embed_dim)

    def forward(self, x, fixed_support, prior_amp=None):
        tokens, skip, grid_size = self.patch_embed(x)
        H, W = grid_size
        tokens_2d = tokens.transpose(1, 2).view(-1, tokens.shape[2], H, W)
        tokens_2d = self.token_refine(tokens_2d)
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        assert tokens.shape[1] == self.pos_embed.shape[1], f"Token shape {tokens.shape[1]} != pos_embed shape {self.pos_embed.shape[1]}"
        x = tokens + self.pos_embed
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)
        predicted_amp, predicted_phase = self.decoder(x, skip, fixed_support, prior_amp)
        cos_phi = predicted_phase[:, 0:1]
        sin_phi = predicted_phase[:, 1:2]
        norm = torch.sqrt(cos_phi ** 2 + sin_phi ** 2 + 1e-8)
        cos_phi = cos_phi / norm
        sin_phi = sin_phi / norm
        predicted_phase = torch.atan2(sin_phi, cos_phi)
        predicted_phase = apply_support_mask(predicted_phase, fixed_support)

        return predicted_amp, predicted_phase

