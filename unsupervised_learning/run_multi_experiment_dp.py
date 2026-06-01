################################################################################
# run_multi_experiment_dp.py
# Batch-run Fourier ViT training and prediction for selected datasets.
# - Jialun Liu, LCN, UCL, 05.2026, jialun.liu.17@ucl.ac.uk
#
# Input:
#   Each dataset folder must contain dp_amp.tif and a support file:
#   support_original.tif or support_vit.tif. phase.tif is optional.
#
# Output:
#   Results are saved to outputs/<dataset_name>/.
################################################################################

import os
import subprocess

# ---------------------- Run mode ----------------------
RUN_MODE = "both"   # "original", "vit", or "both"

if RUN_MODE not in ["original", "vit", "both"]:
    raise ValueError("RUN_MODE must be 'original', 'vit', or 'both'.")

# ---------------------- Run settings ----------------------
NUM_RUNS = 1
NUM_EPOCHS = 1000
FINAL_EPOCH = NUM_EPOCHS - 1

# Each dataset with paths
datasets = {
    "Synthetic-project_regular_10_domain": "../synthetic_dataset/2d_projection/regular_shape/regular_10_domain",
    "Synthetic-project_irregular_10_domain": "../synthetic_dataset/2d_projection/irregular_shape/irregular_10_domain",
    "Synthetic-smooth_amp_10_domain": "../synthetic_dataset/smooth_amplitude/crystal_10_domain",
    "LCMO-500-32-Binning": "../experiment_dataset/LCMO-500-32-Binning",
    "Fe3O4-below_tc": "../experiment_dataset/Fe3O4/below_tc",
    "Fe3O4-above_tc": "../experiment_dataset/Fe3O4/above_tc",
    "Fe3O4-above_tc-vit-support": "../experiment_dataset/Fe3O4/above_tc",
    "Fe3O4-below_tc-vit-support": "../experiment_dataset/Fe3O4/below_tc",
    "STO-1039082-vit-support": "../experiment_dataset/STO-1039082",
}

for name, folder in datasets.items():

    if RUN_MODE == "original" and "vit-support" in name:
        continue

    if RUN_MODE == "vit" and "vit-support" not in name:
        continue

    dp_path = os.path.join(folder, "dp_amp.tif")

    if "vit-support" in name:
        amp_path = os.path.join(folder, "support_vit.tif")
    else:
        amp_path = os.path.join(folder, "support_original.tif")

    phase_path = os.path.join(folder, "phase.tif")  # optional for experiments

    output_dir = f"outputs/{name}"
    os.makedirs(output_dir, exist_ok=True)

    if name == "Synthetic-smooth_amp_10_domain":
        AMP_LOCK_EPOCH = 50
        AMP_PRIOR_DECAY_END = 200
    else:
        AMP_LOCK_EPOCH = 1
        AMP_PRIOR_DECAY_END = 1

    # Set environment variables or pass args
    env = os.environ.copy()
    env["DP_PATH"] = dp_path
    env["AMP_PATH"] = amp_path
    env["NUM_RUNS"] = str(NUM_RUNS)
    env["NUM_EPOCHS"] = str(NUM_EPOCHS)
    env["AMP_LOCK_EPOCH"] = str(AMP_LOCK_EPOCH)
    env["AMP_PRIOR_DECAY_END"] = str(AMP_PRIOR_DECAY_END)
    env["FINAL_EPOCH"] = str(FINAL_EPOCH)

    # MINIMAL CHANGE: handle missing phase.tif gracefully
    if os.path.exists(phase_path):
        env["PHASE_PATH"] = phase_path
    else:
        env["PHASE_PATH"] = ""
        print(f"   (info) No phase.tif found in {folder}; proceeding without GT phase.")

    env["OUTPUT_DIR"] = output_dir

    print(f"\n Dataset: {name}")
    print(f" Support: {amp_path}")
    print(f" Output: {output_dir}")
    print(f" Runs: {NUM_RUNS}")
    print(f" Epochs: {NUM_EPOCHS}")
    print(f" Amplitude prior: lock until epoch {AMP_LOCK_EPOCH}, decay ends at epoch {AMP_PRIOR_DECAY_END}")

    # Call training
    print(f"\n Running training for: {name}")
    subprocess.run(["python", "train_single_input.py"], env=env, check=True)

    # Call prediction
    print(f"\n Running prediction for: {name}")
    subprocess.run(["python", "prediction_single_input.py"], env=env, check=True)