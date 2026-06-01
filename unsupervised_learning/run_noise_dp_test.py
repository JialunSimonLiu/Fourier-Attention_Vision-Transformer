################################################################################
# run_noise_test.py
# Batch-run Fourier ViT noise robustness tests.
# - Jialun Liu, LCN, UCL, 05.2026, jialun.liu.17@ucl.ac.uk
#
# Input:
#   Noisy diffraction TIFFs with shared support_original.tif and optional phase.tif.
#
# Output:
#   Results are saved to outputs/noise_test/<noise_case>/.
################################################################################

import os
import subprocess

# =========================================
# Noise test section
# =========================================
DP_DIR = "../synthetic_dataset/noise_data/2d_projection/noisy_dps"

# ---------------------- Run settings ----------------------
NUM_RUNS = 1
NUM_EPOCHS = 1000
FINAL_EPOCH = NUM_EPOCHS - 1
AMP_LOCK_EPOCH = 1
AMP_PRIOR_DECAY_END = 1

noise_datasets = {
    "low_p":       os.path.join(DP_DIR, "poisson_noisy_magnitude_low.tif"),
    "moderate_p": os.path.join(DP_DIR, "poisson_noisy_magnitude_moderate.tif"),
    "strong_p":   os.path.join(DP_DIR, "poisson_noisy_magnitude_strong.tif"),
    "extreme_p":  os.path.join(DP_DIR, "poisson_noisy_magnitude_extreme.tif"),
    "degraded_p": os.path.join(DP_DIR, "poisson_noisy_magnitude_degraded.tif"),

    "low_g":       os.path.join(DP_DIR, "gaussian_noisy_magnitude_low.tif"),
    "moderate_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_moderate.tif"),
    "strong_g":   os.path.join(DP_DIR, "gaussian_noisy_magnitude_strong.tif"),
    "extreme_g":  os.path.join(DP_DIR, "gaussian_noisy_magnitude_extreme.tif"),
    "degraded_g": os.path.join(DP_DIR, "gaussian_noisy_magnitude_degraded.tif"),

    "low_pc":       os.path.join(DP_DIR, "partial_noisy_magnitude_low.tif"),
    "moderate_pc": os.path.join(DP_DIR, "partial_noisy_magnitude_moderate.tif"),
    "strong_pc":   os.path.join(DP_DIR, "partial_noisy_magnitude_strong.tif"),
    "extreme_pc":  os.path.join(DP_DIR, "partial_noisy_magnitude_extreme.tif"),
    "degraded_pc": os.path.join(DP_DIR, "partial_noisy_magnitude_degraded.tif"),
}

# Fixed reference files for all noise cases
amp_path = os.path.join(DP_DIR, "amp.tif")
phase_path = os.path.join(DP_DIR, "phase.tif")
support_path = os.path.join(DP_DIR, "support_original.tif")

for name, dp_path in noise_datasets.items():

    output_dir = f"outputs/noise_test/{name}"
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["DP_PATH"] = dp_path
    env["AMP_PATH"] = support_path
    env["PHASE_PATH"] = phase_path if os.path.exists(phase_path) else ""
    env["OUTPUT_DIR"] = output_dir

    env["NUM_RUNS"] = str(NUM_RUNS)
    env["NUM_EPOCHS"] = str(NUM_EPOCHS)
    env["AMP_LOCK_EPOCH"] = str(AMP_LOCK_EPOCH)
    env["AMP_PRIOR_DECAY_END"] = str(AMP_PRIOR_DECAY_END)
    env["FINAL_EPOCH"] = str(FINAL_EPOCH)

    print(f"\n Running noise test for: {name}")
    print(f" DP: {dp_path}")
    print(f" Support: {support_path}")
    print(f" Output: {output_dir}")

    # Call training
    subprocess.run(["python", "train_single_input.py"], env=env, check=True)

    # Call prediction
    subprocess.run(["python", "prediction_single_input.py"], env=env, check=True)