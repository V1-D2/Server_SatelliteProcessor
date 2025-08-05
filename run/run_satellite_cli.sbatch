#!/bin/bash
#SBATCH --job-name=satellite_processor_cli
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu  # Adjust based on your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00

echo "============================================"
echo "SatelliteProcessor CLI Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Load required modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Set up Python environment
echo "Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$HOME/satellite_env" ]; then
    echo "Creating virtual environment..."
    python -m venv $HOME/satellite_env
fi

# Activate virtual environment
source $HOME/satellite_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install/update requirements
echo "Installing requirements..."
pip install -r requirements_server.txt

# Set environment variables
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check GPU availability
echo "Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Parse command line arguments
BATCH_MODE=false
FUNCTION=""
DATE=""
ORBIT=""
FILE_INDEX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch)
            BATCH_MODE=true
            shift
            ;;
        --function)
            FUNCTION="$2"
            shift 2
            ;;
        --date)
            DATE="$2"
            shift 2
            ;;
        --orbit)
            ORBIT="$2"
            shift 2
            ;;
        --file-index)
            FILE_INDEX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Starting SatelliteProcessor CLI"
echo "Mode: $([ "$BATCH_MODE" = true ] && echo "Batch" || echo "Interactive")"

if [ "$BATCH_MODE" = true ]; then
    echo "Function: $FUNCTION"
    echo "Date: $DATE"
    echo "Orbit: $ORBIT"
    echo "File Index: $FILE_INDEX"

    # Run in batch mode
    python main_cli.py \
        --batch \
        --function "$FUNCTION" \
        --date "$DATE" \
        ${ORBIT:+--orbit "$ORBIT"} \
        ${FILE_INDEX:+--file-index "$FILE_INDEX"}
else
    # Run in interactive mode
    python main_cli.py
fi

EXIT_CODE=$?

echo "============================================"
echo "SatelliteProcessor CLI Job Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE