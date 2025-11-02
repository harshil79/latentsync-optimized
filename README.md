# LatentSync – Optimized

This repository contains the modified version of **LatentSync**, updated and optimized for CUDA, CPU & MacBook hardware using MPS acceleration.  
The project implements an **adaptive per-frame guidance mechanism** based on short-time audio energy and multiple environment-level optimizations to achieve improved runtime efficiency while maintaining lip-sync quality.

---

## Environment Setup

The environment setup is automated using `setup-env.sh`.  
This script installs all dependencies, handles OS-level libraries, and downloads required checkpoints.

### How to Run the Setup
```bash
git clone https://github.com/harshil79/latentsync-optimized.git

chmod +x setup-env.sh

source setup-env.sh
```

This will:

- Create a new conda environment named latentsync with Python 3.10.13
- Install required libraries and dependencies (ffmpeg, opencv, torch, diffusers, etc.)
- Install HuggingFace CLI if not already installed
- Download the model checkpoints:
    - whisper/tiny.pt
    - latentsync_unet.pt

- Verify that OpenCV and ffmpeg are installed correctly

### Note on OS Compatibility
- On macOS (Apple Silicon M1/M2/M3), OpenCV is installed using Homebrew.
- On Linux, the system dependency libgl1 is installed automatically using the `setu-env.sh`.

# Part 1: Environment and Pipeline Optimizations

## A. Environment Setup Optimization

During the initial setup, the original env-setup script provided by the base repository was strictly CUDA-oriented and failed to run correctly on MPS devices.
I have replaced and optimized the setup process to improve compatibility and runtime on different devices.

### This version was limited by:

- CUDA-only logic (no fallback for CPU/MPS)
- Manual dependencies for OpenCV
- No environment verification or fault tolerance
- No conditional system handling (macOS/Linux)
- Redundant huggingface-cli command for model download

### Latest version handles:

- Compatibility on non-GPU devices
- Prevention of runtime missing-library errors
- Requirement.txt with latest/appropriate libraries (e.g. eva-decord for python>=3.10)
- Easier debugging and clean setup output

## B .Inference speedup optimizations

### Profiling & benchmarking
#### All benchmarks were performed on a MacBook Pro M3 (8-core CPU, 10-core GPU, 16GB RAM) using MPS backend

A short profiling run was conducted using:

- Input frames: 8
- Resolution: 128×128
- Denoising steps: 10 per inference
- Test video: demo1 (9s, female, outdoor conditions, average brightness)

Original pipeline Inference test results

- Affine Transformation time (242 iterations): 27.4 s
- Inference time 31 Itr (8 frames, Res - 128p, 10 denoising steps): 11162.7s (~361s/it)
- Memory usage (average GB): 8.1
- CPU Utilization: ~55%
- Total runtime: 3h 5m

Optimized pipeline test results

- Affine Transformation time (242 iterations): 22.6 s
- Inference time 31 Itr (8 frames, Res - 128p, 10 denoising steps): 7130.4s (~230s/it)
- Memory usage (average GB): 5.3
- CPU Utilization: ~87%
- Total runtime: 2h 1m

### Optimization details

List of the optimizations I implemented for speedup, with their benefits clearly stated:

a. Added targeted FP16 autocast for VAE encode/decode and UNet forward passes
- Enabled mixed precision only inside VAE encode/decode and UNet forward blocks instead of full-pipeline casting.
- Benefit: Reduced compute time and memory transfers on MPS without quality degradation.

b. Implemented CPU fallback for VAE operations on MPS OOM
- Added explicit error handling for "MPS backend" and "out of memory" exceptions to retry encode/decode on CPU.
- Benefit: Prevented crashes and full model reloads, maintaining smooth runtime on memory-limited GPUs and MPS devices.

c. Replaced .view() with .reshape() and enforced .contiguous() in Whisper feature slicing
- Fixed invalid tensor stride issues during audio embedding preparation.
- Benefit: Removed tensor copy overhead and stabilized preprocessing (~2× faster feature slicing).

d. Parallelized data preparation using ThreadPoolExecutor
- Split CPU-heavy steps (mask creation, latent prep, affine transforms) into background threads while GPU processed inference.
- Benefit: Improved pipeline throughput by overlapping I/O and compute.

c. Enabled fused attention kernels for UNet
- Activated PyTorch’s built-in Flash/Memory-Efficient attention backends.
- Benefit: Reduced attention computation latency per diffusion step.

d. Added automatic device/provider selection (get_device() and ONNX provider logic)
- Unified CPU, CUDA, and MPS handling across modules.
- Benefit: Eliminated manual configuration and ensured optimal backend execution on each system.

## C. Overall improvement per inference step:
- Affine Transformation time (242 iterations): ~17.5%
- Inference time 31 Itr (8 frames, Res - 128p, 10 denoising steps): ~36%
- Memory usage (average GB): ~17.5%
- CPU Utilization: ~-32% (increased CPU utilization due to hardware limitations and parallell data processing)
- Total runtime: ~34%