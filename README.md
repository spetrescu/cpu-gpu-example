# Assignment 1: CPU vs GPU Image Reconstruction (`09.04` & `14.04.2026`)

## Overview
In this assignment, you are given a set of transformed (obfuscated) image frames. Your task is to reconstruct the original frames by implementing the inverse of a reversible image-processing pipeline written in PyTorch.

Unlike the first lab, you are not given the original images. Instead, you must recover them by carefully reversing the sequence of operations that were applied during the forward pass.

After reconstruction, you should also combine the frames into a video and compare execution performance between CPU and GPU.

---

## Repository structure
```
.
├── frames/                        # Input frames (already transformed)
├── forward_batch.py               # Forward pipeline (provided)
├── inverse_batch_exercise.py      # Your task (contains the TODOs)
└── README.md
```
---
The `forward_batch.py` file was used to obfuscate the images -- you do not have to run it -- it is there to provide explanation on how the forward sequence of transformations was conducted.

## Setup

### 1. Clone (or pull) repository
```
git clone https://github.com/spetrescu/cpu-gpu-example.git
```
or, if you have the repo already, pull with `git pull`.

### 2. Create a virtual environment
```
python3 -m venv venv
```

Activate it:
```
source venv/bin/activate
```

---

### 3. Install dependencies
```
pip install numpy opencv-python torch Pillow
```

---

## Your Task

You must complete the TODOs in: `inverse_batch_exercise.py`. Subsequently, you must run the file (see below guidance on how to run it and with which flags) and collect measurements for CPU vs CUDA enabled acceleration.

---

## Expected Outcome

After completing the implementation, your code should: (1) reconstruct the original frames exactly, (2) produce visually correct images, and (3) allow comparison of compute between CPU and GPU.

You are expected to:

- Measure runtime for CPU and GPU execution
- Produce plots comparing performance

As the save time (I/O) dominates the compute time for the transformations, make sure you only focus on the compute time for the visualizations (`when using the parameter --save_outputs you will see that the save time, i.e., writing the images, is separated from the actual compute`).

Specifically, after you have implemented the TODOs, you must make comparsions (by varying the different batch sizes). Here is an example below of running your implementation with a `batch_size` of 2 both on CPU and GPU.
```
python inverse_batch_exercise.py \
  --input_dir frames \
  --output_dir out_reconstructed_cpu \
  --recipe recipe.json \
  --device cpu \
  --batch_size 2 \
  --repeat 1
```
```
python inverse_batch_exercise.py \
  --input_dir frames \
  --output_dir out_reconstructed_cuda \
  --recipe recipe.json \
  --device cuda \
  --batch_size 2 \
  --repeat 1
```

---
After you have implemented the TODOs and successfully (1) completed collecting measurements for at least 5 different batch sizes, (2) created a plot with comparisons between CPU and GPU performance, and (3) created a montage of the video -- you can consider the assignment as passed.

## Instructions for the first demo, hands-on session (07.04.2026)
The goal is to reconstruct the image frames that are obfuscated (cd into `first_session_07_04_2026`).
1. Create venv: `python3 -m venv lab-1-gpu`
2. Activate using: `source lab-1-gpu/bin/activate`
3. Install deps: `pip install numpy opencv-python `
4. Run extract frames command: `python extract_frames.py cpu_video.mp4 cpu_frames`

5. python cpu_demo_frames.py make-demo cpu_frames cpu_demo_input

6. Run obfuscation test:
```
python cpu_demo_frames.py forward cpu_demo_input cpu_demo_obfuscated \
  --manifest cpu_demo_manifest.json \
  --levels 64,32,16 --rounds 6 --seed 1337 --frame-permute
```

7. CPU reverse transformations
```
python cpu_demo_frames.py reverse cpu_demo_obfuscated cpu_demo_reconstructed \
  --manifest cpu_demo_manifest.json
```

8. Verify that the frames were correctly formatted using:
```
python cpu_demo_frames.py verify cpu_demo_input cpu_demo_reconstructed
```

## GPUs now come in strong as a point of comparison
1. Install the missing dependency: `pip install torch`
2. Run the equivalent compression script now on the GPU:
```
python gpu_demo_frames.py forward cpu_demo_input cpu_demo_obfuscated_gpu \
  --manifest cpu_demo_gpu_manifest.json \
  --levels 64,32,16 --rounds 256 --seed 1337 --frame-permute --batch-size 3
```
3. Run the reverse of the transformations of a GPU:
```
python gpu_demo_frames.py reverse cpu_demo_obfuscated_gpu cpu_demo_reconstructed_gpu \
  --manifest cpu_demo_gpu_manifest.json --batch-size 3
```
4. Verify things are identical: `python gpu_demo_frames.py verify cpu_demo_input cpu_demo_reconstructed_gpu`
