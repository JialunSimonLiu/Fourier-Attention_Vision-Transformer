Unsupervised Fourier ViT Workflow
=================================

Author
------
Jialun Liu, LCN, UCL, 06.2026
jialun.liu.17@ucl.ac.uk


Overview
--------
This folder runs unsupervised Fourier ViT phase retrieval for synthetic and experimental BCDI diffraction patterns.

Model:
    fourier_vit_model.py

Main workflow:
    train_single_input.py
    prediction_single_input.py

Batch scripts:
    run_multi_experiment_dp.py
    run_noise_dp_test.py


How to run
----------
From the unsupervised_learning folder:

    python run_multi_experiment_dp.py

For noise tests:

    python run_noise_dp_test.py


Input
-----
Each dataset folder should contain:

    dp_amp.tif
    support_original.tif or support_vit.tif
    phase.tif    optional

The scripts read data from:

    ../synthetic_dataset/
    ../experiment_dataset/


Main datasets
-------------
run_multi_experiment_dp.py currently uses:

    Synthetic-project_regular_10_domain
    Synthetic-project_irregular_10_domain
    Synthetic-smooth_amp_10_domain
    LCMO-500-32-Binning
    Fe3O4-below_tc
    Fe3O4-above_tc
    Fe3O4-above_tc-vit-support
    Fe3O4-below_tc-vit-support
    STO-1039082-vit-support


Support mode
------------
Set this in run_multi_experiment_dp.py:

    RUN_MODE = "original"   # use support_original.tif
    RUN_MODE = "vit"        # use support_vit.tif
    RUN_MODE = "both"       # run both groups

Datasets ending with -vit-support use support_vit.tif.
All other datasets use support_original.tif.


Run settings
------------
Set the number of runs and epochs in run_multi_experiment_dp.py:

    NUM_RUNS = 2
    NUM_EPOCHS = 1000
    FINAL_EPOCH = NUM_EPOCHS - 1

For example:

    NUM_RUNS = 20

runs 20 independent reconstructions.


Amplitude-prior schedule
------------------------
The amplitude-prior blending is controlled by:

    AMP_LOCK_EPOCH
    AMP_PRIOR_DECAY_END

For the smooth-amplitude synthetic case:

    AMP_LOCK_EPOCH = 50
    AMP_PRIOR_DECAY_END = 200

For projected synthetic and experimental cases:

    AMP_LOCK_EPOCH = 1
    AMP_PRIOR_DECAY_END = 1

Meaning:

    before AMP_LOCK_EPOCH:
        use the prior amplitude

    between AMP_LOCK_EPOCH and AMP_PRIOR_DECAY_END:
        blend from prior amplitude to learned amplitude

    after AMP_PRIOR_DECAY_END:
        use the learned amplitude


Output
------
Results are saved to:

    outputs/<dataset_name>/

Noise-test results are saved to:

    outputs/noise_test/<noise_case>/

Each output folder contains checkpoints, prediction results, chi-square histogram, and radial diffraction profile.


Note
----
The workflow is unsupervised. The model is trained by diffraction-space consistency, not by ground-truth real-space amplitude or phase labels.
