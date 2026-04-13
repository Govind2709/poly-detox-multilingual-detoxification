#!/bin/bash
# ============================================================================
# SLURM job script: Poly-Detox on cluster
# ============================================================================
# Usage:
#   sbatch run_notebook.sh
#
# First-time setup (run ONCE manually before sbatch):
#   conda create -n polydetox python=3.12 -y
#   conda activate polydetox
#   pip install -r ~/poly_detox/requirements.txt
# ============================================================================

#SBATCH --job-name=poly-detox
#SBATCH --output=poly_detox_%j.log
#SBATCH --error=poly_detox_%j.err
#SBATCH --partition=Teaching           # Edinburgh Informatics Teaching partition
#SBATCH --nodelist=saxa                # H200 node specifically
#SBATCH --gres=gpu:1                   # request 1 GPU (mT5-base fits in ~8 GB)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00              # 2 days (Teaching partition max)

# ── Activate conda properly ─────────────────────────────────────────────────
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate polydetox

# ── Set working directory (absolute path) ─────────────────────────────────
cd "$HOME/poly_detox"

echo "=== Job started at $(date) ==="
echo "=== Node: $(hostname) ==="
nvidia-smi
echo "=== Python: $(which python) ==="
echo "=== Torch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())') ==="
echo ""

# ── Force single GPU (prevents DataParallel + LoRA + NCCL Error 5) ────────
export CUDA_VISIBLE_DEVICES=0

# ── Clean outputs only — preserve data_cache (datasets + mT5 weights) ───────
# data_cache is kept to avoid re-downloading on every run (~several GB).
# Delete it manually if you need a truly fresh environment.
echo "=== Cleaning checkpoints, results, figures (keeping data_cache) ==="
rm -rf output/poly_detox/checkpoints/
rm -rf output/poly_detox/results/
rm -rf output/poly_detox/figures/
rm -rf output/poly_detox/pairs/
rm -rf output/poly_detox/inspect/
echo "=== Clean done ==="

# Run as plain Python script (faster than nbconvert, same code)
python run_all.py

echo ""
echo "=== Job finished at $(date) ==="
echo "=== Results in: $(pwd)/output/poly_detox/ ==="
