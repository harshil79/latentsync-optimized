#!/bin/bash -e

# Step 1: Check for conda
if ! command -v conda &> /dev/null; then
    echo "Conda not found." 
    echo "Install Miniconda from https://docs.conda.io/en/latest/miniconda.html and try again."
    exit 1
fi

# Step 2: Create conda environment
ENV_NAME="latentsync"
conda create -y -n $ENV_NAME python=3.10.13

echo "Environment '$ENV_NAME' created."
conda activate $ENV_NAME

# Step 3: Install ffmpeg
conda install -y -n $ENV_NAME -c conda-forge ffmpeg

# Step 4: Install Python dependencies
conda run -n $ENV_NAME pip install -r requirements.txt

# Step 5: Install OpenCV system dependency
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y libgl1
    else
        echo "apt not found."
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        brew install opencv
    else
        echo "Homebrew not found."
        exit 1
    fi
else
    echo "OS not recognized. Install OpenCV system dependencies manually."
fi

# Step 6: Install huggingface-cli if not present
if ! command -v huggingface-cli &> /dev/null; then
    conda run -n $ENV_NAME pip install huggingface_hub
fi

# Step 7: Download checkpoints
mkdir -p checkpoints
conda run -n $ENV_NAME hf download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
conda run -n $ENV_NAME hf download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

echo "Environment setup complete"