################################################################################
# Run multiple synthetic diffraction pattern with train and prediction files
# - Jialun Liu, LCN, UCL, 11-12.2025, jialun.liu.17@ucl.ac.uk
################################################################################

import os
import subprocess
"""
datasets = {
    "LCMO-500": "Jialun_test_data/LCMO-500",
    "LR17-53": "Jialun_test_data/LR17-53",
    "STO-1039082": "Jialun_test_data/STO-1039082",
    "NEL-25": "Jialun_test_data/NEL-25",
    "LR17-68": "Jialun_test_data/LR17-68",
    "Sythetic": "Jialun_test_data/Sythetic",
    "LCMO-500-NEW": "Jialun_test_data/LCMO-500-NEW",
    "LR17-53-NEW": "Jialun_test_data/LR17-53-NEW",
    "LR17-68-NEW": "Jialun_test_data/LR17-68-NEW",
    "NEL-43-NEW": "Jialun_test_data/NEL-43-NEW",
    "STO-1039082": "Jialun_test_data/STO-1039082",
    "Sythetic-10": "Jialun_test_data/Sythetic-10",
    "Sythetic-15": "Jialun_test_data/Sythetic-15",
    "Sythetic-19": "Jialun_test_data/Sythetic-19",
    "LCMO-500-38": "Jialun_test_data/LCMO-500-38",
    "LCMO-500-32": "Jialun_test_data/LCMO-500-32",
    "LCMO-500-32-Binning": "Jialun_test_data/LCMO-500-32-Binning",
    "LCMO-500-32-Binning+4": "Jialun_test_data/LCMO-500-32-Binning+4",
    "LCMO-500-32-Binning-4": "Jialun_test_data/LCMO-500-32-Binning-4",
}
"""
# Each dataset with paths
datasets = {
    "LCMO-500-32-Binning-final": "Jialun_test_data/LCMO-500-32-Binning",
    #"LCMO-500-32-Binning+4": "Jialun_test_data/LCMO-500-32-Binning+4",
    #"STO-1039082": "Jialun_test_data/STO-1039082",
}

for name, folder in datasets.items():
    print(f"\n🔹 Running training for: {name}")

    dp_path = os.path.join(folder, "dp_amp.tif")
    amp_path = os.path.join(folder, "support_original.tif")
    phase_path = os.path.join(folder, "phase.tif")  # optional for experiments

    output_dir = f"outputs/{name}"
    os.makedirs(output_dir, exist_ok=True)

    # Set environment variables or pass args
    env = os.environ.copy()
    env["DP_PATH"] = dp_path
    env["AMP_PATH"] = amp_path
    # MINIMAL CHANGE: handle missing phase.tif gracefully
    if os.path.exists(phase_path):
        env["PHASE_PATH"] = phase_path
    else:
        env["PHASE_PATH"] = ""
        print(f"   (info) No phase.tif found in {folder}; proceeding without GT phase.")
    env["OUTPUT_DIR"] = output_dir

    # Call training
    #subprocess.run(["python", "train_single_input.py"], env=env, check=True)
    #subprocess.run(["python", "train_single_input_simulation.py"], env=env, check=True)

    # Call prediction
    print(f"\n🔹 Running prediction for: {name}")
    subprocess.run(["python", "prediction_single_input.py"], env=env, check=True)
