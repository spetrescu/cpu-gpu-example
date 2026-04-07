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