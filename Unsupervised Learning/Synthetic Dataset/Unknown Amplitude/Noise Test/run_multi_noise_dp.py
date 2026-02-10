################################################################################
# Run multiple synthetic diffraction pattern with train and prediction files
# - Jialun Liu, LCN, UCL, 08-10.2025, jialun.liu.17@ucl.ac.uk
################################################################################
import os
import subprocess

# Run from this folder to avoid cwd surprises
ROOT = os.path.dirname(os.path.abspath(__file__))
DP_DIR = os.path.join(ROOT, "noisy_dps")

# --- Fixed crystal (same for ALL noisy DPs) ---
GT_AMP_PATH   = os.path.join(DP_DIR, "gt_amp_1.tif")
GT_PHASE_PATH = os.path.join(DP_DIR, "gt_phase_1.tif")
"""
datasets = {
    "low_p":      os.path.join(DP_DIR, "poisson_noisy_magnitude_low.tif"),
    "moderate_p": os.path.join(DP_DIR, "poisson_noisy_magnitude_moderate.tif"),
    "strong_p":   os.path.join(DP_DIR, "poisson_noisy_magnitude_strong.tif"),
    "extreme_p":  os.path.join(DP_DIR, "poisson_noisy_magnitude_extreme.tif"),
    "degraded_p": os.path.join(DP_DIR, "poisson_noisy_magnitude_degraded.tif"),
    "low_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_low.tif"),
    "moderate_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_moderate.tif"),
    "strong_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_strong.tif"),
    "extreme_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_extreme.tif"),
    "degraded_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_degraded.tif"),
    "low_pc":      os.path.join(DP_DIR, "partial_noisy_magnitude_low.tif"),
    "moderate_pc": os.path.join(DP_DIR, "partial_noisy_magnitude_moderate.tif"),
    "strong_pc":   os.path.join(DP_DIR, "partial_noisy_magnitude_strong.tif"),
    "extreme_pc":  os.path.join(DP_DIR, "partial_noisy_magnitude_extreme.tif"),
    "degraded_pc": os.path.join(DP_DIR, "partial_noisy_magnitude_degraded.tif"),
}
"""
datasets = {
    "low_p":      os.path.join(DP_DIR, "poisson_noisy_magnitude_low.tif"),
    "moderate_p": os.path.join(DP_DIR, "poisson_noisy_magnitude_moderate.tif"),
    "strong_p":   os.path.join(DP_DIR, "poisson_noisy_magnitude_strong.tif"),
    "extreme_p":  os.path.join(DP_DIR, "poisson_noisy_magnitude_extreme.tif"),
    "degraded_p": os.path.join(DP_DIR, "poisson_noisy_magnitude_degraded.tif"),
    "low_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_low.tif"),
    "moderate_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_moderate.tif"),
    "strong_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_strong.tif"),
    "extreme_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_extreme.tif"),
    "degraded_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_degraded.tif"),
    "low_pc":      os.path.join(DP_DIR, "partial_noisy_magnitude_low.tif"),
    "moderate_pc": os.path.join(DP_DIR, "partial_noisy_magnitude_moderate.tif"),
    "strong_pc":   os.path.join(DP_DIR, "partial_noisy_magnitude_strong.tif"),
    "extreme_pc":  os.path.join(DP_DIR, "partial_noisy_magnitude_extreme.tif"),
    "degraded_pc": os.path.join(DP_DIR, "partial_noisy_magnitude_degraded.tif"),
}

# Sanity checks: make failures obvious
for label, path in datasets.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{label}] DP not found: {path}")
for k, p in [("GT_AMP_PATH", GT_AMP_PATH), ("GT_PHASE_PATH", GT_PHASE_PATH)]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{k} not found: {p}")

for name, dp_path in datasets.items():
    print(f"\n Running training for: {name}")

    # Always use the same crystal for all noisy DPs
    amp_path   = GT_AMP_PATH
    phase_path = GT_PHASE_PATH

    output_dir = os.path.join(ROOT, "outputs", name)
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["DP_PATH"]    = dp_path
    env["AMP_PATH"]   = amp_path
    env["PHASE_PATH"] = phase_path
    env["OUTPUT_DIR"] = output_dir
    env["CLEAN_DP_PATH"] = os.path.join(DP_DIR, "gt_dp_mag_1.tif")

    # Train
    subprocess.run(["python", "train_noise.py"], env=env, check=True, cwd=ROOT)

    # Predict / Evaluate
    #print(f"\n Running prediction for: {name}")
    subprocess.run(["python", "prediction_noise.py"], env=env, check=True, cwd=ROOT)

# collecting results
subprocess.run(["python", "result_paper.py"], check=True, cwd=ROOT)

print("\n All datasets finished.")
