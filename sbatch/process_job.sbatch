#!/bin/bash
#SBATCH --job-name=satproc_job
#SBATCH --output=/home/vdidur/Server_SatelliteProcessor/logs/%x_%j.out
#SBATCH --error=/home/vdidur/Server_SatelliteProcessor/logs/%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=2:00:00

echo "============================================"
echo "SatelliteProcessor Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# Change to project directory
cd /home/vdidur/Server_SatelliteProcessor

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH=/home/vdidur/Server_SatelliteProcessor:$PYTHONPATH

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the job processor
python scripts/job_processor.py

echo "============================================"
echo "SatelliteProcessor Job Finished: $(date)"
echo "============================================"