import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_video")
    ap.add_argument("output_dir")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = outdir / f"frame_{count:06d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Failed to write frame: {frame_path}")
        count += 1

    cap.release()

    meta = {
        "source_video": str(Path(args.input_video).resolve()),
        "fps": fps,
        "frame_count": count,
        "width": width,
        "height": height,
        "format": "png_sequence_bgr",
        "pattern": "frame_%06d.png",
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Extracted {count} frames to {outdir}")
    print(f"FPS: {fps:.6f}")
    print(f"Resolution: {width}x{height}")
    print(f"Wrote: {outdir / 'metadata.json'}")


if __name__ == "__main__":
    main()
