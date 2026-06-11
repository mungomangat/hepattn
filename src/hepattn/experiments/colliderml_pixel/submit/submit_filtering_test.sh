#!/bin/bash

#SBATCH --job-name=colliderml-pixel-filter-test
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=4:00:00
#SBATCH --output=/share/rcif2/mmangat/slurm_logs/slurm-%j.%x.out

source /share/rcif2/mmangat/.env
unset COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE: $COMET_WORKSPACE"

echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Update these to point at the desired run and checkpoint
RUN_DIR="/share/rcif2/mmangat/logs/colliderml_pixel_filtering/colliderml-pixel-filtering_20260601-T000240"
CKPT_PATH="${RUN_DIR}/ckpts/epoch=036-val_loss=0.59060.ckpt"
CONFIG_PATH="${RUN_DIR}/config.yaml"

TEST_DIR="/share/rcif2/mmangat/data/colliderml/ttbar/test/"

cd /share/rcif2/mmangat/hepattn-worktrees/colliderML/src/hepattn/experiments/colliderml_pixel/
echo "Moved dir, now in: ${PWD}"

export TMPDIR=/var/tmp/

echo "nvidia-smi:"
nvidia-smi

echo "Running inference on test split..."
PYTORCH_CMD="python run_filtering.py test --config ${CONFIG_PATH} --ckpt_path ${CKPT_PATH} --data.test_dir ${TEST_DIR}"
PIXI_CMD="pixi run $PYTORCH_CMD"
APPTAINER_CMD="srun apptainer run --nv --bind /share/rcifdata/ --bind /share/rcif2/ /share/rcif2/mmangat/hepattn-worktrees/colliderML/pixi.sif $PIXI_CMD"
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD

echo "Done! HDF5 written to ${RUN_DIR}/ckpts/"
