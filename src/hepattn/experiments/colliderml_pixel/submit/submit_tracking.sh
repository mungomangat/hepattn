#!/bin/bash

#SBATCH --job-name=colliderml-pixel-tracking
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=/share/rcif2/mmangat/slurm_logs/slurm-%j.%x.out

source /share/rcif2/mmangat/.env

unset COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE: $COMET_WORKSPACE"

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Move to experiment directory
cd /share/rcif2/mmangat/hepattn-worktrees/colliderML/src/hepattn/experiments/colliderml_pixel/
echo "Moved dir, now in: ${PWD}"

# Use a per-worktree TMPDIR to avoid conflicts with concurrent Slurm jobs
export TMPDIR=/var/tmp/

echo "nvidia-smi:"
nvidia-smi

echo "Running training script..."

PYTORCH_CMD="python run_tracking.py fit --config configs/tracking.yaml --trainer.devices 1"

PIXI_CMD="pixi run $PYTORCH_CMD"

APPTAINER_CMD="srun apptainer run --nv --bind /share/rcifdata/ --bind /share/rcif2/ /share/rcif2/mmangat/hepattn-worktrees/colliderML/pixi.sif $PIXI_CMD"

echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
