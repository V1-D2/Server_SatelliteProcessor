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
echo "Memory allocated: 64GB per GPU"
echo "============================================"

# Set environment variables to suppress nvidia-smi warnings
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Change to project directory
cd /home/vdidur/Server_SatelliteProcessor

# Install required packages if not already installed
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user h5py pyproj matplotlib pillow scipy xarray tqdm paramiko numpy opencv-python gportal

# Test environment in TensorFlow container
echo "Testing environment in TensorFlow container:"
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ NumPy: {e}')
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU count: {torch.cuda.device_count()}')
        print(f'✅ GPU name: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')
try:
    import h5py
    print(f'✅ h5py: {h5py.__version__}')
except Exception as e:
    print(f'❌ h5py: {e}')
try:
    import pyproj
    print(f'✅ pyproj: {pyproj.__version__}')
except Exception as e:
    print(f'❌ pyproj: {e}')
try:
    import matplotlib
    print(f'✅ matplotlib: {matplotlib.__version__}')
except Exception as e:
    print(f'❌ matplotlib: {e}')
"

echo "============================================"
echo "Checking project structure..."
ls -la /home/vdidur/Server_SatelliteProcessor | head -10
echo "Jobs pending: $(ls /home/vdidur/Server_SatelliteProcessor/jobs/pending/*.json 2>/dev/null | wc -l)"

echo "============================================"
echo "Starting SatelliteProcessor Job Processing:"

# Set Python path
export PYTHONPATH=/home/vdidur/Server_SatelliteProcessor:$PYTHONPATH

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the job processor in TensorFlow container
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/Server_SatelliteProcessor:/home/vdidur/Server_SatelliteProcessor \
    --env PYTHONPATH=/home/vdidur/Server_SatelliteProcessor:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python /home/vdidur/Server_SatelliteProcessor/scripts/job_processor.py

# Check if job processing was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Job processing completed successfully!"
    echo "Results saved in: /home/vdidur/Server_SatelliteProcessor/results/"
    echo ""
    echo "Completed jobs:"
    ls -la /home/vdidur/Server_SatelliteProcessor/jobs/completed/ | tail -5
else
    echo "============================================"
    echo "ERROR: Job processing failed!"
    echo "Check the error messages above"
    echo "Failed jobs:"
    ls -la /home/vdidur/Server_SatelliteProcessor/jobs/failed/ | tail -5
fi

echo "============================================"
echo "SatelliteProcessor Job Finished: $(date)"
echo "============================================"