# cpu-gpu-example
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