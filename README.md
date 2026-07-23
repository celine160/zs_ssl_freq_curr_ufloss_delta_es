# ZS-SSL: Zero-Shot Self-Supervised Learning Extension

## ZS-SSL Overview
ZS-SSL enables physics-guided deep learning MRI reconstruction using only a single slice/sample ([paper](https://openreview.net/forum?id=085y6YPaYjP)).
Succintly, ZS-SSL  partitions the available measurements from a single scan into three disjoint sets. Two of these sets are used to enforce data consistency and define loss during training for self-supervision, while the last set serves to self-validate, establishing an early stopping criterion. In the presence of models pre-trained on a database with different image characteristics, ZS-SSL can be combined with transfer learning (TL) for faster convergence time and reduced computational complexity.

<img src="figs/zs_ssl_overview.PNG" align="center" width="750px"> <br>

*An overview of the proposed zero-shot self-supervised learning approach. a) Acquired
measurements for the single scan are partitioned into three sets: a training (Θ) and loss mask (Λ) for
self-supervision, and a self-validation mask for automated early stopping (Γ). b) The parameters,
θ, of the unrolled MRI reconstruction network are updated using Θ and Λ in the data consistency
(DC) units of the unrolled network and for defining loss, respectively. c) Concurrently, a k-space
validation procedure is used to establish the stopping criterion by using Ω\Γ in the DC units and Γ
to measure a validation loss. d) Once the network training has been stopped due to an increasing
trend in the k-space validation loss, the final reconstruction is performed using the relevant learned
network parameters and all the acquired measurements in the DC unit.*

---

## **New Extensions: Frequency Curriculum & UFLoss**

This repository has been extended to support several robust training enhancements:

### 1. Frequency Curriculum Training
Instead of calculating the loss uniformly across k-space from the start, we progressively reveal higher frequencies over time. This acts as a curriculum learning technique, stabilizing the self-supervised training phase. 
- Enable via `--use_frequency_curriculum`
- The `K` (width of the initial center crop) expands incrementally throughout training based on `--frequency_curriculum_decay`.

### 2. UFLoss (Unsupervised Feature Loss)
To enhance high-frequency detail and edge sharpness, an Unsupervised Feature Loss network can be integrated into the objective function alongside the standard L1 loss.
- Provide the path to the UFLoss checkpoint via `--ufloss_path <path/to/checkpoint>`
- Control the weighting of the loss via `--ufloss_weight` (e.g., `0.5`).

### 3. Early Stopping on Validation Delta (Delta-ES)
In addition to stopping when validation loss stops decreasing, we've implemented early stopping based on the **difference** between training loss and validation loss, ensuring the model doesn't overfit to the self-supervised masks.
- Enable via `--stop_on_validation_delta`
- Adjust strictness with `--delta_es_tightness` (higher = stops sooner, less prone to overfitting).

---

## Installation
Dependencies are given in environment.yml. A new conda environment can be installed with
```
conda env create -f environment.yaml
```

## Datasets
We have used the [fastMRI](https://fastmri.med.nyu.edu/) dataset in our experiments.

## How to use

### End-to-End Evaluation Pipeline
We provide a convenient bash script to run the Baseline, Frequency Curriculum, and UFLoss methods sequentially on a single `.mat` file:

```bash
./run_single_test.sh data/path_to_your_scan.mat
```
This script will automatically format the output directories and run the three configurations side-by-side.

### Visualizing Reconstructions & Metrics
Once a test sequence completes, you can visualize and compare the final reconstructions side-by-side using the automated plotting script:

```bash
python plot_best_recons.py --base_dir saved_models_file_brain_... --data_file data/path_to_your_scan.mat
```
This generates a cleanly formatted `all_models_recons_comparison.png` displaying the Target, Baseline, Frequency Curriculum, and UFLoss reconstructions along with x10 boosted error maps. A zoomed ROI plot (`zoomed_recons_comparison.png`) is also automatically generated.

### Plotting Early Stopping Curves
To analyze the validation and loss curves across epochs (including the Delta gap):
```bash
python plot_early_stopping.py --base_dir saved_models_file_brain_...
```
This yields `early_stopping_metrics_plot.png`.

---
