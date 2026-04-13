# Poly-Detox: Multilingual Text Detoxification with LoRA-Adapted mT5
**Group G001** — MLP Coursework 4 (2025–26)

---

## Overview

This repository contains the implementation of **Poly-Detox**, a parameter-efficient
multilingual text detoxification system built on mT5-base fine-tuned with Low-Rank
Adaptation (LoRA). The system targets English (EN), Spanish (ES), and Hindi (HI) and
is evaluated across five experiments (A–E) covering baselines, English LoRA training,
zero-shot cross-lingual transfer, few-shot adaptation, and LoRA hyperparameter ablation.

---

## File Structure

```
G001_source_code/
├── run_all.py                  # Main experiment script (all 5 experiments A–E)
├── generate_figures.py         # Generates all report figures from saved result JSONs
├── check_backtranslation.py    # Validates NLLB backtranslation data augmentation
├── run_notebook.sh             # SLURM job submission script (UoE Teaching cluster)
├── requirements.txt            # Python dependencies
├── poly_detox_complete.ipynb   # Jupyter notebook with full execution outputs
└── README.md                   # This file
```

---

## Hardware

All experiments were run on the **University of Edinburgh MLP Teaching Cluster**:

| Resource | Specification |
|---|---|
| Node | `saxa` |
| GPU | NVIDIA H200 (80 GB HBM3) |
| CPU cores | 4 |
| RAM | 16 GB |
| Partition | `Teaching` |
| Max walltime | 2 days (`2-00:00:00`) |

Peak GPU memory during training: **16.65 GB** (few-shot, Exp D).
Total wall-clock time for all experiments: **~9 hours**.

---

## Environment Setup

Perform this **once** on the cluster before submitting the job:

```bash
# SSH into the cluster
ssh <UUN>@mlp.inf.ed.ac.uk

# Create and activate conda environment
conda create -n polydetox python=3.12 -y
conda activate polydetox

# Upload source code to home directory
# (from local machine, or clone/copy to ~/poly_detox/)

# Install dependencies
pip install -r ~/poly_detox/requirements.txt
```

> **Note:** The first run downloads ~10 GB of data and model weights
> (mT5-base, XLM-R toxicity classifier, LaBSE, NLLB-200, XGLM) via
> HuggingFace Hub into `output/poly_detox/data_cache/`.
> Subsequent runs reuse the cache automatically.

---

## Running on the Cluster

### Submit via SLURM (recommended)

```bash
cd ~/poly_detox
sbatch run_notebook.sh
```

Monitor the job:

```bash
squeue -u $USER                        # check job status
tail -f poly_detox_<jobid>.log         # live stdout log
cat  poly_detox_<jobid>.err            # error log
```

### Run interactively (debugging only)

```bash
srun --partition=Teaching --nodelist=saxa --gres=gpu:1 \
     --cpus-per-task=4 --mem=32G --pty bash

conda activate polydetox
cd ~/poly_detox
export CUDA_VISIBLE_DEVICES=0
python run_all.py
```

---

## What `run_all.py` Does

The script runs all five experiments sequentially:

| Experiment | Description | Output |
|---|---|---|
| **A** | Delete & Identity baselines on EN/ES/HI | `results/exp_a_*.json` |
| **B** | English LoRA training (`r=32, α=64, Q+V`) | `checkpoints/english_lora/` |
| **C** | Zero-shot cross-lingual transfer to ES/HI | `results/exp_c_*.json` |
| **D** | Few-shot adaptation (50/100/200 shots) | `results/exp_d_*.json` |
| **E** | LoRA hyperparameter ablation sweep | `results/exp_e_*.json` |

Results are saved as JSON to `output/poly_detox/results/`.
Qualitative output pairs are saved to `output/poly_detox/pairs/`.

---

## Generating Figures

After `run_all.py` completes, generate all report figures:

```bash
python generate_figures.py
```

Figures are saved to `output/poly_detox/figures/`.

---

## Output Directory Structure

```
output/poly_detox/
├── checkpoints/        # Trained LoRA adapter weights (NOT included in submission)
├── results/            # Evaluation JSON files per experiment
├── figures/            # Generated PNG figures
├── pairs/              # Qualitative input/output pair text files
└── data_cache/         # HuggingFace dataset and model cache (NOT included)
```

> Model weights and data are excluded from this submission per coursework guidelines.

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `google/mt5-base` (970M params) |
| LoRA rank `r` | 32 |
| LoRA alpha `α` | 64 |
| LoRA target modules | Q + V projections |
| Trainable parameters | 3,538,944 (0.36%) |
| Batch size (effective) | 16 (2 per device × 8 grad accum steps) |
| Learning rate (EN) | 3×10⁻⁴ |
| Learning rate (few-shot) | 1×10⁻⁴ |
| Precision | fp16 mixed precision |
| EN training time | ~41 min |
