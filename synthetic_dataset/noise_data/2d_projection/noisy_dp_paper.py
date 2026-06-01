################################################################################
# noisy_dp.py
# Noise simulations for 2D Bragg diffraction magnitude patterns (DP),
# generating Poisson/Gaussian/partial-coherence corrupted variants
# - Jialun Liu, LCN, UCL, 28.10.2025, jialun.liu.17@ucl.ac.uk
# Input: GT DP magnitude .tif (normalised here to [0,1], intensity = mag^2)
# Output: noisy magnitude TIFFs (uint16), panel plots, and TXT metadata
# - Poisson branch simulates photon counting statistics using a fixed scale S
# - Gaussian branch adds additive noise in count space (sigma scaled by S)
# - Partial coherence branch uses Gaussian blur in intensity with flux preserved
# - Model input normalisation uses sqrt(counts/C_scale) -> magnitude in [0,1]
################################################################################

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import argparse
from scipy.ndimage import gaussian_filter

# === CLI Args ===
parser = argparse.ArgumentParser(description="DP noise")
parser.add_argument('--input', type=str, default="dp_amp.tif", help='Input magnitude .tif, normalized here to [0,1]')
parser.add_argument('--output_dir', type=str, default="noisy_dps", help='Output directory')
parser.add_argument('--save_individual_magnitude', action='store_true', help='If set, also save per-variant magnitude PNGs')
parser.add_argument('--save_histograms', action='store_true', help='If set, also save per-variant magnitude histograms')
args = parser.parse_args()

# === Setup ===
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

np.random.seed(1)

# === Noise Levels (fixed) ===
noise_levels = [
    {"label": "low",      "poisson_scale": 1100, "gauss_sigma": 0.001, "partial_sigma": 0.36},
    {"label": "moderate", "poisson_scale": 300, "gauss_sigma": 0.0015, "partial_sigma": 0.44},
    {"label": "strong",   "poisson_scale": 100,  "gauss_sigma": 0.0025, "partial_sigma": 0.48},
    {"label": "extreme",  "poisson_scale": 50,  "gauss_sigma": 0.004, "partial_sigma": 0.55},
    {"label": "degraded", "poisson_scale": 15,  "gauss_sigma": 0.0055, "partial_sigma": 0.62},
]

NORM_MODE = "max"
PERCENTILE = 99.9

# === Functions ===
def add_poisson_noise_only(intensity_norm, poisson_scale):
    expected_counts = np.clip(intensity_norm * float(poisson_scale), 0.0, None).astype(np.float32)
    return np.random.poisson(expected_counts).astype(np.float32)

def add_gaussian_noise_only(intensity_norm, gaussian_level_counts, poisson_scale):
    expected_counts = np.clip(intensity_norm * float(poisson_scale), 0.0, None).astype(np.float32)
    gaussian_noise = np.random.normal(0.0, float(gaussian_level_counts), expected_counts.shape).astype(np.float32)
    return np.clip(expected_counts + gaussian_noise, 0.0, None).astype(np.float32)

def apply_partial_coherence_blur(intensity_norm, sigma):
    # Gaussian blur (beam coherence)
    blurred = gaussian_filter(intensity_norm, sigma=sigma, mode='reflect')
    # Preserve total integrated flux
    blurred *= np.sum(intensity_norm) / (np.sum(blurred) + 1e-8)
    # Renormalize to same max as input (prevents flattening)
    blurred /= np.max(blurred) + 1e-8
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)

def normalize_counts_for_model(counts, S, mode="max", percentile=99.9):
    """
    Return (M_model, C_scale) where M_model is magnitude in [0,1] for the model input.
    We normalize photon counts, then convert to magnitude by sqrt.
    """
    if mode == "expected":
        C_scale = float(S)
    elif mode == "p999":
        C_scale = float(np.percentile(counts, percentile))
    else:  # "max"
        C_scale = float(np.max(counts))
    C_scale = max(C_scale, 1e-8)
    I_model = counts / C_scale
    M_model = np.sqrt(np.clip(I_model, 0.0, None)).astype(np.float32)
    return np.clip(M_model, 0.0, 1.0), C_scale

def calculate_chi_square(predicted, target):
    """
    "chi_square" monitor (normalized squared error with max-norm scaling).
    Same for training/plots.
    """
    predicted_norm = predicted / (np.max(predicted) + 1e-8)
    target_norm = target / (np.max(target) + 1e-8)
    numerator = np.sum((predicted_norm - target_norm)**2)
    denominator = np.sqrt(np.sum(predicted_norm**2) * np.sum(target_norm**2) + 1e-8)
    return numerator / denominator

def calculate_pcc(predicted, target):
    predicted_norm = predicted / (np.max(predicted) + 1e-8)
    target_norm = target / (np.max(target) + 1e-8)
    mean_pred = np.mean(predicted_norm)
    mean_target = np.mean(target_norm)
    numerator = np.sum((predicted_norm - mean_pred) * (target_norm - mean_target))
    denominator = np.sqrt(np.sum((predicted_norm - mean_pred)**2) * (np.sum((target_norm - mean_target)**2)) + 1e-8)
    return numerator / denominator

def pearson_chi_square_poisson(observed_counts, expected_counts):
    """
    Pearson (Poisson) chi-square: sum((obs - exp)^2 / exp).
    Returns (chi2_total, chi2_reduced).
    """
    exp = np.clip(expected_counts.astype(np.float64), 1e-8, None)
    obs = observed_counts.astype(np.float64)
    chi2 = np.sum((obs - exp) ** 2 / exp)
    dof = obs.size - 1
    chi2_red = chi2 / max(dof, 1)
    return float(chi2), float(chi2_red)

def compute_relative_snr(signal, noisy):
    signal_power = np.mean(signal**2)
    noise_power = np.mean((noisy - signal)**2)
    return signal_power / (noise_power + 1e-8)

def snr_db(signal, noisy, eps=1e-12):
    s = np.asarray(signal, dtype=np.float64)
    n = np.asarray(noisy,  dtype=np.float64)
    sp = np.mean(s*s)
    npow = np.mean((n - s)**2)
    return 10.0 * np.log10((sp + eps) / (npow + eps))

def plot_histograms(label, clean_mag, noisy_mag):
    plt.figure(figsize=(10,4))
    plt.hist(clean_mag.flatten(), bins=100, alpha=0.6, label='Clean Magnitude')
    plt.hist(noisy_mag.flatten(), bins=100, alpha=0.6, label='Noisy Magnitude')
    plt.legend()
    plt.title(f"Histogram of Magnitude: {label}")
    plt.xlabel("Magnitude"); plt.ylabel("Pixel Count")
    plt.tight_layout()
    plt.savefig(f"plots/hist_mag_{label}.png", dpi=300)
    plt.close()

def plot_magnitude(label, mag):
    plt.imshow(np.clip(mag, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
    plt.title(f"DP Magnitude (linear): {label}")
    plt.colorbar(label='Magnitude')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"plots/magnitude_dp_{label}.png", dpi=300)
    plt.close()

def plot_panel_poisson_vs_gaussian(clean_mag, poisson_mag_dict, gaussian_mag_dict, order_labels):
    """2 rows (Poisson top, Gaussian bottom), columns = GT + noise levels, with axis ticks & labels."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------- basics --------------------
    ncols = 1 + len(order_labels)  # +1 for GT
    fig, axs = plt.subplots(2, ncols, figsize=(4*ncols, 8))
    H, W = clean_mag.shape
    xticks = np.linspace(0, W-1, 5, dtype=int)
    yticks = np.linspace(0, H-1, 5, dtype=int)

    # Helper to apply consistent axis cosmetics
    def _style_axis(ax, is_bottom_row, is_left_col):
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', which='both', labelsize=10)
        # Only show labels on outer axes to avoid clutter
        if not is_bottom_row:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Qx (px)", fontsize=12)

        if not is_left_col:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Qy (px)", fontsize=12)

    # Ensure output folder exists
    os.makedirs("plots", exist_ok=True)

    # -------------------- Leftmost column = GT --------------------
    for r in range(2):
        ax = axs[r, 0]
        im = ax.imshow(np.clip(clean_mag, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
        ax.set_title("GT Magnitude" if r == 0 else "", fontsize=12)

        _style_axis(ax, is_bottom_row=(r == 1), is_left_col=True)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Intensity (a.u.)", fontsize=11)

    # -------------------- Other columns = noise levels --------------------
    for j, lbl in enumerate(order_labels, start=1):
        # ---- Poisson (top row) ----
        ax_p = axs[0, j]
        M_p = poisson_mag_dict.get(lbl, None)
        if M_p is None:
            ax_p.axis("off")
            ax_p.set_title(f"Poisson – {lbl}", fontsize=12)
        else:
            im_p = ax_p.imshow(np.clip(M_p, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
            ax_p.set_title(f"Poisson – {lbl}", fontsize=12)
            _style_axis(ax_p, is_bottom_row=False, is_left_col=(j == 0))
            cbar = plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Intensity (a.u.)", fontsize=11)

        # ---- Gaussian (bottom row) ----
        ax_g = axs[1, j]
        M_g = gaussian_mag_dict.get(lbl, None)
        if M_g is None:
            ax_g.axis("off")
            ax_g.set_title(f"Gaussian – {lbl}", fontsize=12)
        else:
            im_g = ax_g.imshow(np.clip(M_g, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
            ax_g.set_title(f"Gaussian – {lbl}", fontsize=12)
            _style_axis(ax_g, is_bottom_row=True, is_left_col=(j == 0))
            cbar = plt.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Intensity (a.u.)", fontsize=11)

    # Optional row labels on the very left (GT column y-labels already say Qy)
    axs[0, 0].set_ylabel("Qy (px)\n(Poisson)", fontsize=12)
    axs[1, 0].set_ylabel("Qy (px)\n(Gaussian)", fontsize=12)

    plt.suptitle("DP Magnitude Comparison: Poisson vs Gaussian (normalized to [0,1])", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.savefig("plots/panel_poisson_vs_gaussian.png", dpi=300)
    plt.close()

def plot_panel_one_row_four(clean_mag, poisson_mag_dict, gaussian_mag_dict):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    titles_imgs = [
        ("Poisson – low",      poisson_mag_dict.get("low", None)),
        ("Poisson – degraded", poisson_mag_dict.get("degraded", None)),
        ("Gaussian – low",     gaussian_mag_dict.get("low", None)),
        ("Gaussian – degraded",gaussian_mag_dict.get("degraded", None)),
    ]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.8))

    H, W = clean_mag.shape
    xticks = [0, 16, 32, 48, W-1]
    yticks = [0, 16, 32, 48, H-1]
    tick_labels = ["0", "16", "32", "48", "64"]  # show 64 instead of 63

    def _style_axis(ax, is_left_col):
        ax.set_xticks(xticks); ax.set_yticks(yticks)
        ax.set_xticklabels(tick_labels, fontsize=15)
        if is_left_col:
            ax.set_yticklabels(tick_labels, fontsize=15)
            ax.set_ylabel("Qy (px)", fontsize=20)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        ax.set_xlabel("Qx (px)", fontsize=20)
        ax.tick_params(axis='both', which='both', labelsize=15)

    for i, (title, img) in enumerate(titles_imgs):
        ax = axs[i]
        if img is None:
            ax.axis("off")
            continue

        im = ax.imshow(np.clip(img, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
        #ax.set_title(title, fontsize=20)
        _style_axis(ax, is_left_col=(i == 0))

        # per-axis colorbar, same look as your 2×N panel
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Intensity (a.u.)", fontsize=15)

    # identical layout pattern to your reference function
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("plots/panel_noise_one_row.png", dpi=300)
    plt.close()

def plot_panel_poisson_gaussian_partial(clean_mag, poisson_mag_dict, gaussian_mag_dict, partial_mag_dict, order_labels):
    """
    3 rows (Poisson, Gaussian, Partial-Coherence) × columns = GT + noise levels.
    Same style as plot_panel_poisson_vs_gaussian, extended for partial coherence.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    ncols = 1 + len(order_labels)  # +1 for GT
    nrows = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 12))

    H, W = clean_mag.shape
    xticks = np.linspace(0, W-1, 5, dtype=int)
    yticks = np.linspace(0, H-1, 5, dtype=int)

    def _style_axis(ax, is_bottom_row, is_left_col):
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', which='both', labelsize=10)
        if not is_bottom_row:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Qx (px)", fontsize=12)

        if not is_left_col:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Qy (px)", fontsize=12)

    os.makedirs("plots", exist_ok=True)

    # --- Leftmost GT column ---
    for r in range(nrows):
        ax = axs[r, 0]
        im = ax.imshow(np.clip(clean_mag, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
        if r == 0:
            ax.set_title("GT Magnitude", fontsize=12)
        _style_axis(ax, is_bottom_row=(r == nrows-1), is_left_col=True)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Intensity (a.u.)", fontsize=11)

    # --- Noise columns ---
    for j, lbl in enumerate(order_labels, start=1):
        # Row 0: Poisson
        ax_p = axs[0, j]
        M_p = poisson_mag_dict.get(lbl, None)
        if M_p is not None:
            im_p = ax_p.imshow(np.clip(M_p, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
            ax_p.set_title(f"Poisson – {lbl}", fontsize=12)
            _style_axis(ax_p, is_bottom_row=False, is_left_col=(j == 0))
            cbar = plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Intensity (a.u.)", fontsize=11)
        else:
            ax_p.axis("off")

        # Row 1: Gaussian
        ax_g = axs[1, j]
        M_g = gaussian_mag_dict.get(lbl, None)
        if M_g is not None:
            im_g = ax_g.imshow(np.clip(M_g, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
            ax_g.set_title(f"Gaussian – {lbl}", fontsize=12)
            _style_axis(ax_g, is_bottom_row=False, is_left_col=(j == 0))
            cbar = plt.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Intensity (a.u.)", fontsize=11)
        else:
            ax_g.axis("off")

        # Row 2: Partial Coherence
        ax_pc = axs[2, j]
        M_pc = partial_mag_dict.get(lbl, None)
        if M_pc is not None:
            im_pc = ax_pc.imshow(np.clip(M_pc, 0.0, 1.0), cmap='turbo', vmin=0, vmax=1)
            ax_pc.set_title(f"Partial – {lbl}", fontsize=12)
            _style_axis(ax_pc, is_bottom_row=True, is_left_col=(j == 0))
            cbar = plt.colorbar(im_pc, ax=ax_pc, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Intensity (a.u.)", fontsize=11)
        else:
            ax_pc.axis("off")

    axs[0, 0].set_ylabel("Qy (px)\n(Poisson)", fontsize=12)
    axs[1, 0].set_ylabel("Qy (px)\n(Gaussian)", fontsize=12)
    axs[2, 0].set_ylabel("Qy (px)\n(Partial)", fontsize=12)

    plt.suptitle("DP Magnitude Comparison: Poisson vs Gaussian vs Partial-Coherence", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plots/panel_poisson_gaussian_partial.png", dpi=300)
    plt.close()

def write_metadata_txt(path, d):
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")

# === Load magnitude, normalize to [0,1] ===
dp_mag = tiff.imread(args.input).astype(np.float32)
dp_mag /= np.max(dp_mag) + 1e-8
clean_intensity = dp_mag ** 2  # I_clean in [0,1]

# Optional: GT magnitude individual plot
if args.save_individual_magnitude:
    plot_magnitude("GT", dp_mag)

poisson_mag_by_label = {}
gaussian_mag_by_label = {}
partial_mag_by_label = {}

for setting in noise_levels:
    label = setting["label"]
    sigma = setting["gauss_sigma"]
    S = float(setting["poisson_scale"])
    print(f"Noise: {label} (σ={sigma})")

    # --- Poisson-only branch ---
    counts_p = add_poisson_noise_only(clean_intensity, S)

    # Model input (unified normalization for training/ViT)
    M_p, C_scale_p = normalize_counts_for_model(counts_p, S, mode=NORM_MODE, percentile=PERCENTILE)
    poisson_mag_by_label[label] = M_p
    tiff.imwrite(f"{args.output_dir}/poisson_noisy_magnitude_{label}.tif", (M_p * 65535).astype(np.uint16))

    # Evaluation on physical S-scale (fair metrics vs GT)
    I_p_eval = counts_p / (S + 1e-8)
    M_p_eval = np.sqrt(I_p_eval).astype(np.float32)

    chi_p_I = calculate_chi_square(M_p_eval ** 2, dp_mag ** 2)
    pcc_p_I = calculate_pcc(M_p_eval ** 2, dp_mag ** 2)
    chi_p_M = calculate_chi_square(M_p_eval, dp_mag)
    pcc_p_M = calculate_pcc(M_p_eval, dp_mag)
    snr_p_I = snr_db(dp_mag**2, M_p_eval**2)
    snr_p_M = snr_db(dp_mag, M_p_eval)
    is_high_noise_p = "yes" if chi_p_M > 0.20 else "no"

    # Pearson (Poisson) χ² in COUNTS space (for reporting)
    expected_counts = S * clean_intensity
    pearson_chi2_total, pearson_chi2_reduced = pearson_chi_square_poisson(counts_p, expected_counts)

    print(f"[Poisson] Chi²_I={chi_p_I:.5f}, PCC_I={pcc_p_I:.5f}, SNR_db_I={snr_p_I:.2f}")
    print(f"[Poisson] Chi²_M={chi_p_M:.5f}, PCC_M={pcc_p_M:.5f}, SNR_db_M={snr_p_M:.2f}")
    print(f"[Poisson] Pearson χ² (counts): total={pearson_chi2_total:.2f}, reduced={pearson_chi2_reduced:.6f}")

    metadata_p = {
        "label": f"poisson_{label}",
        "poisson_scale": setting["poisson_scale"],
        "gauss_sigma": 0.0,
        "chi_square_I": float(chi_p_I),
        "pcc_I": float(pcc_p_I),
        "snr_db_I": float(snr_p_I),
        "chi_square_M": float(chi_p_M),
        "pcc_M": float(pcc_p_M),
        "snr_db_M": float(snr_p_M),
        "pearson_chi2_total_counts": float(pearson_chi2_total),
        "pearson_chi2_reduced_counts": float(pearson_chi2_reduced),
        "max_magnitude": float(np.max(M_p)),  # stats of the saved (model-input) magnitude
        "mean_magnitude": float(np.mean(M_p)),
        "scale_mode": NORM_MODE,
        "count_scale_value": float(C_scale_p),
    }
    write_metadata_txt(f"metadata/metadata_poisson_{label}.txt", metadata_p)

    if args.save_histograms:
        plot_histograms(f"poisson_{label}", dp_mag, M_p)
    if args.save_individual_magnitude:
        plot_magnitude(f"poisson_{label}", M_p)

    # --- Gaussian-only branch ---
    counts_g = add_gaussian_noise_only(clean_intensity, sigma * S, S)

    # Model input (same unified normalization policy)
    M_g, C_scale_g = normalize_counts_for_model(counts_g, S, mode=NORM_MODE, percentile=PERCENTILE)
    gaussian_mag_by_label[label] = M_g
    tiff.imwrite(f"{args.output_dir}/gaussian_noisy_magnitude_{label}.tif", (M_g * 65535).astype(np.uint16))

    # Evaluation on physical S-scale
    I_g_eval = counts_g / (S + 1e-8)
    M_g_eval = np.sqrt(np.clip(I_g_eval, 0.0, None)).astype(np.float32)

    chi_g_I = calculate_chi_square(M_g_eval**2, dp_mag**2)
    pcc_g_I = calculate_pcc(M_g_eval**2, dp_mag**2)
    chi_g_M = calculate_chi_square(M_g_eval, dp_mag)
    pcc_g_M = calculate_pcc(M_g_eval, dp_mag)
    snr_g_I = snr_db(dp_mag**2, M_g_eval**2)
    snr_g_M = snr_db(dp_mag, M_g_eval)
    is_high_noise_g = "yes" if chi_g_M > 0.20 else "no"

    print(f"[Gaussian] Chi²_I={chi_g_I:.5f}, PCC_I={pcc_g_I:.5f}, SNR_db_I={snr_g_I:.2f}")
    print(f"[Gaussian] Chi²_M={chi_g_M:.5f}, PCC_M={pcc_g_M:.5f}, SNR_db_M={snr_g_M:.2f}")

    metadata_g = {
        "label": f"gaussian_{label}",
        "poisson_scale": 0.0,
        "gauss_sigma": sigma,
        "chi_square_I": float(chi_g_I),
        "pcc_I": float(pcc_g_I),
        "snr_db_I": float(snr_g_I),
        "chi_square_M": float(chi_g_M),
        "pcc_M": float(pcc_g_M),
        "snr_db_M": float(snr_g_M),
        "max_magnitude": float(np.max(M_g)),
        "mean_magnitude": float(np.mean(M_g)),
        "scale_mode": NORM_MODE,
        "count_scale_value": float(C_scale_g),
    }
    write_metadata_txt(f"metadata/metadata_gaussian_{label}.txt", metadata_g)

    if args.save_histograms:
        plot_histograms(f"gaussian_{label}", dp_mag, M_g)
    if args.save_individual_magnitude:
        plot_magnitude(f"gaussian_{label}", M_g)

    # --- Partial-Coherence branch ---
    sigma_pc = setting["partial_sigma"]
    I_pc = apply_partial_coherence_blur(clean_intensity, sigma_pc)

    # Model input normalization (same policy as others)
    M_pc, C_scale_pc = normalize_counts_for_model(I_pc * S, S, mode=NORM_MODE, percentile=PERCENTILE)
    partial_mag_by_label[label] = M_pc
    tiff.imwrite(f"{args.output_dir}/partial_noisy_magnitude_{label}.tif", (M_pc * 65535).astype(np.uint16))

    # Evaluation on physical S-scale
    I_pc_eval = I_pc
    M_pc_eval = np.sqrt(I_pc_eval).astype(np.float32)

    chi_pc_I = calculate_chi_square(M_pc_eval**2, dp_mag**2)
    pcc_pc_I = calculate_pcc(M_pc_eval**2, dp_mag**2)
    chi_pc_M = calculate_chi_square(M_pc_eval, dp_mag)
    pcc_pc_M = calculate_pcc(M_pc_eval, dp_mag)
    snr_pc_I = snr_db(dp_mag**2, M_pc_eval**2)
    snr_pc_M = snr_db(dp_mag, M_pc_eval)

    print(f"[PartialCoherence] σ_pc={sigma_pc:.2f} Chi²_I={chi_pc_I:.5f}, PCC_I={pcc_pc_I:.5f}, SNR_db_I={snr_pc_I:.2f}")
    print(f"[PartialCoherence] σ_pc={sigma_pc:.2f} Chi²_M={chi_pc_M:.5f}, PCC_M={pcc_pc_M:.5f}, SNR_db_M={snr_pc_M:.2f}")

    metadata_pc = {
        "label": f"partial_{label}",
        "poisson_scale": 0.0,
        "gauss_sigma": sigma,
        "partial_sigma": sigma_pc,
        "coherence_length_px": float(1.0 / (sigma_pc + 1e-8)),
        "chi_square_I": float(chi_pc_I),
        "pcc_I": float(pcc_pc_I),
        "snr_db_I": float(snr_pc_I),
        "chi_square_M": float(chi_pc_M),
        "pcc_M": float(pcc_pc_M),
        "snr_db_M": float(snr_pc_M),
        "max_magnitude": float(np.max(M_pc)),
        "mean_magnitude": float(np.mean(M_pc)),
        "scale_mode": NORM_MODE,
        "count_scale_value": float(C_scale_pc),
    }

    write_metadata_txt(f"metadata/metadata_partial_{label}.txt", metadata_pc)

    if args.save_histograms:
        plot_histograms(f"partial_{label}", dp_mag, M_pc)
    if args.save_individual_magnitude:
        plot_magnitude(f"partial_{label}", M_pc)


# 2-row panel only
order = [nl["label"] for nl in noise_levels]
#plot_panel_poisson_vs_gaussian(dp_mag, poisson_mag_by_label, gaussian_mag_by_label, order)

#print("Saved panel: plots/panel_poisson_vs_gaussian.png\nTXT metadata saved in metadata/")

# One-row, four-panel export (Poisson/Gaussian × Low/Degraded)
plot_panel_one_row_four(dp_mag, poisson_mag_by_label, gaussian_mag_by_label)
print("Saved panel: plots/panel_noise_one_row.png\nTXT metadata saved in metadata/")

# --- 3-row panel with Poisson / Gaussian / Partial-Coherence ---
plot_panel_poisson_gaussian_partial(
    dp_mag, poisson_mag_by_label, gaussian_mag_by_label, partial_mag_by_label, order)
print("Saved panel: plots/panel_poisson_gaussian_partial.png\nTXT metadata saved in metadata/")

