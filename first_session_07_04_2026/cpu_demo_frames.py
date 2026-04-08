
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# numpy opencv-python 

def parse_levels(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def read_metadata(frame_dir: Path) -> Dict:
    with open(frame_dir / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata(frame_dir: Path, meta: Dict):
    with open(frame_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def list_frames(frame_dir: Path) -> List[Path]:
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No PNG frames found in {frame_dir}")
    return frames


def load_frame_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_frame_rgb(path: Path, arr: np.ndarray):
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write {path}")


def copy_selected_frames(src_dir: Path, dst_dir: Path, indices: List[int], fps: float):
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_frames = list_frames(src_dir)
    for out_idx, src_idx in enumerate(indices):
        arr = load_frame_rgb(src_frames[src_idx])
        save_frame_rgb(dst_dir / f"frame_{out_idx:06d}.png", arr)

    meta = {
        "source_frames_dir": str(src_dir.resolve()),
        "fps": fps,
        "frame_count": len(indices),
        "selected_indices": indices,
        "pattern": "frame_%06d.png",
        "format": "png_sequence_rgb",
    }
    write_metadata(dst_dir, meta)


def stream_idx(x: int) -> int:
    return int(x)


def _rng_for(base_seed: int, level_idx: int, round_idx: int, stream_id: int, frame_idx: int = 0):
    seed = (
        int(base_seed) * 0x9E3779B1
        + level_idx * 0x85EBCA6B
        + round_idx * 0xC2B2AE35
        + stream_idx(stream_id) * 0x27D4EB2F
        + frame_idx * 0x165667B1
    ) & 0xFFFFFFFF
    return np.random.RandomState(seed)


def _make_perm(n: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n, dtype=perm.dtype)
    return perm, inv


def pad_frame_to_levels(frame: np.ndarray, levels: List[int]) -> Tuple[np.ndarray, Dict]:
    h, w, _ = frame.shape
    max_tile = max(levels)
    pad_h = (max_tile - (h % max_tile)) % max_tile
    pad_w = (max_tile - (w % max_tile)) % max_tile
    padded = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, {"orig_h": h, "orig_w": w, "pad_h": pad_h, "pad_w": pad_w}


def unpad_frame(frame: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    return frame[:orig_h, :orig_w, :]


def _tiles_view(img: np.ndarray, tile: int) -> np.ndarray:
    h, w, c = img.shape
    rows, cols = h // tile, w // tile
    return img.reshape(rows, tile, cols, tile, c).transpose(0, 2, 1, 3, 4)


def _apply_tile_permutation_forward(img: np.ndarray, tile: int, perm: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    rows, cols = h // tile, w // tile
    tiles = _tiles_view(img, tile).reshape(rows * cols, tile, tile, c)
    out = tiles[perm].reshape(rows, cols, tile, tile, c)
    return out.transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def _apply_tile_permutation_inverse(img: np.ndarray, tile: int, inv: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    rows, cols = h // tile, w // tile
    tiles = _tiles_view(img, tile).reshape(rows * cols, tile, tile, c)
    out = tiles[inv].reshape(rows, cols, tile, tile, c)
    return out.transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def _apply_in_tile_perm_forward(img: np.ndarray, tile: int, perm_pix: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    rows, cols = h // tile, w // tile
    tiles = _tiles_view(img, tile).reshape(rows * cols, tile * tile, c)
    out = tiles[:, perm_pix, :].reshape(rows, cols, tile, tile, c)
    return out.transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def _apply_in_tile_perm_inverse(img: np.ndarray, tile: int, inv_pix: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    rows, cols = h // tile, w // tile
    tiles = _tiles_view(img, tile).reshape(rows * cols, tile * tile, c)
    out = tiles[:, inv_pix, :].reshape(rows, cols, tile, tile, c)
    return out.transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def transform_frame_forward(frame: np.ndarray, levels: List[int], rounds: int, seed: int, frame_idx: int) -> np.ndarray:
    cur = frame.copy()
    for li, tile in enumerate(levels):
        for ri in range(rounds):
            total_tiles = (cur.shape[0] // tile) * (cur.shape[1] // tile)

            rng_tiles = _rng_for(seed, li, ri, 101, frame_idx)
            perm_tiles, _ = _make_perm(total_tiles, rng_tiles)
            cur = _apply_tile_permutation_forward(cur, tile, perm_tiles)

            rng_pix = _rng_for(seed, li, ri, 202, frame_idx)
            perm_pix, _ = _make_perm(tile * tile, rng_pix)
            cur = _apply_in_tile_perm_forward(cur, tile, perm_pix)

    return cur


def transform_frame_reverse(frame: np.ndarray, levels: List[int], rounds: int, seed: int, frame_idx: int) -> np.ndarray:
    cur = frame.copy()
    for li_rev, tile in enumerate(reversed(levels)):
        li = len(levels) - 1 - li_rev
        for ri_rev in range(rounds):
            ri = rounds - 1 - ri_rev

            rng_pix = _rng_for(seed, li, ri, 202, frame_idx)
            _, inv_pix = _make_perm(tile * tile, rng_pix)
            cur = _apply_in_tile_perm_inverse(cur, tile, inv_pix)

            total_tiles = (cur.shape[0] // tile) * (cur.shape[1] // tile)
            rng_tiles = _rng_for(seed, li, ri, 101, frame_idx)
            _, inv_tiles = _make_perm(total_tiles, rng_tiles)
            cur = _apply_tile_permutation_inverse(cur, tile, inv_tiles)

    return cur


def build_frame_permutation(num_frames: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState((seed * 0xA24BAED3 + 0x9FB21C65) & 0xFFFFFFFF)
    return _make_perm(num_frames, rng)


def make_demo(src_dir: Path, dst_dir: Path):
    _ = read_metadata(src_dir)
    frames = list_frames(src_dir)
    n = len(frames)
    indices = sorted(set([0, n // 2, n - 1]))
    fps = 1.0

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    copy_selected_frames(src_dir, dst_dir, indices, fps)
    print(f"Created 3-frame demo at {dst_dir}")
    print(f"Selected indices: {indices}")


def forward(input_dir: Path, output_dir: Path, manifest_path: Path, levels: List[int], rounds: int, seed: int, frame_permute: bool):
    meta = read_metadata(input_dir)
    frame_paths = list_frames(input_dir)
    frames = [load_frame_rgb(p) for p in frame_paths]

    padded = []
    pad_info = None
    for fr in frames:
        p, info = pad_frame_to_levels(fr, levels)
        padded.append(p)
        if pad_info is None:
            pad_info = info

    t0 = time.time()
    frame_times = []
    out = []

    for i, fr in enumerate(padded):
        t_frame = time.time()
        print(f"[CPU] forward frame {i+1}/{len(padded)} ...", end="", flush=True)

        out_frame = transform_frame_forward(fr, levels, rounds, seed, i)
        out.append(out_frame)

        dt = time.time() - t_frame
        frame_times.append(dt)
        print(f" done in {dt:.2f}s")

    transform_seconds = time.time() - t0

    if frame_permute:
        perm, inv = build_frame_permutation(len(out), seed)
        out = [out[i] for i in perm]
    else:
        perm = np.arange(len(out))
        inv = perm.copy()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, fr in enumerate(out):
        save_frame_rgb(output_dir / f"frame_{i:06d}.png", fr)

    out_meta = dict(meta)
    out_meta["frame_count"] = len(out)
    out_meta["fps"] = float(meta.get("fps", 1.0))
    write_metadata(output_dir, out_meta)

    manifest = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "fps": float(meta.get("fps", 1.0)),
        "frame_count": len(out),
        "levels": levels,
        "rounds": rounds,
        "seed": seed,
        "frame_permute": bool(frame_permute),
        "frame_perm": perm.tolist(),
        "frame_inv": inv.tolist(),
        "orig_h": int(pad_info["orig_h"]),
        "orig_w": int(pad_info["orig_w"]),
        "pad_h": int(pad_info["pad_h"]),
        "pad_w": int(pad_info["pad_w"]),
        "cpu_forward_seconds": transform_seconds,
        "cpu_forward_frame_seconds": frame_times,
        "cpu_forward_avg_frame_seconds": float(transform_seconds / len(out)) if out else 0.0,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {output_dir}")
    print(f"Wrote {manifest_path}")
    print(f"CPU forward total time: {transform_seconds:.2f}s")
    if out:
        print(f"CPU forward average time per frame: {transform_seconds / len(out):.2f}s")


def reverse(input_dir: Path, output_dir: Path, manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    meta = read_metadata(input_dir)
    frame_paths = list_frames(input_dir)
    frames = [load_frame_rgb(p) for p in frame_paths]

    if manifest.get("frame_permute", False):
        inv = np.array(manifest["frame_inv"], dtype=np.int64)
        frames = [frames[i] for i in inv]

    levels = [int(x) for x in manifest["levels"]]
    rounds = int(manifest["rounds"])
    seed = int(manifest["seed"])

    t0 = time.time()
    frame_times = []
    recovered = []

    for i, fr in enumerate(frames):
        t_frame = time.time()
        print(f"[CPU] reverse frame {i+1}/{len(frames)} ...", end="", flush=True)

        out = transform_frame_reverse(fr, levels, rounds, seed, i)
        out = unpad_frame(out, int(manifest["orig_h"]), int(manifest["orig_w"]))
        recovered.append(out)

        dt = time.time() - t_frame
        frame_times.append(dt)
        print(f" done in {dt:.2f}s")

    transform_seconds = time.time() - t0

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, fr in enumerate(recovered):
        save_frame_rgb(output_dir / f"frame_{i:06d}.png", fr)

    out_meta = dict(meta)
    out_meta["fps"] = float(manifest["fps"])
    out_meta["frame_count"] = len(recovered)
    write_metadata(output_dir, out_meta)

    manifest["cpu_reverse_seconds"] = transform_seconds
    manifest["cpu_reverse_frame_seconds"] = frame_times
    manifest["cpu_reverse_avg_frame_seconds"] = float(transform_seconds / len(recovered)) if recovered else 0.0

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {output_dir}")
    print(f"CPU reverse total time: {transform_seconds:.2f}s")
    if recovered:
        print(f"CPU reverse average time per frame: {transform_seconds / len(recovered):.2f}s")


def verify(dir_a: Path, dir_b: Path):
    a = list_frames(dir_a)
    b = list_frames(dir_b)

    if len(a) != len(b):
        print(f"Frame count mismatch: {len(a)} vs {len(b)}")
        raise SystemExit(1)

    mismatches = 0
    for i, (pa, pb) in enumerate(zip(a, b)):
        ia = load_frame_rgb(pa)
        ib = load_frame_rgb(pb)
        if ia.shape != ib.shape or not np.array_equal(ia, ib):
            mismatches += 1
            diff = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
            print(f"Mismatch on frame {i}: max abs diff={diff.max()}")

    if mismatches == 0:
        print("VERIFY PASSED: all frames identical.")
    else:
        print(f"VERIFY FAILED: mismatched frames={mismatches}")
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s0 = sub.add_parser("make-demo")
    s0.add_argument("input_dir")
    s0.add_argument("output_dir")

    s1 = sub.add_parser("forward")
    s1.add_argument("input_dir")
    s1.add_argument("output_dir")
    s1.add_argument("--manifest", required=True)
    s1.add_argument("--levels", default="64,32,16")
    s1.add_argument("--rounds", type=int, default=256)
    s1.add_argument("--seed", type=int, default=1337)
    s1.add_argument("--frame-permute", action="store_true")

    s2 = sub.add_parser("reverse")
    s2.add_argument("input_dir")
    s2.add_argument("output_dir")
    s2.add_argument("--manifest", required=True)

    s3 = sub.add_parser("verify")
    s3.add_argument("dir_a")
    s3.add_argument("dir_b")

    args = ap.parse_args()

    if args.cmd == "make-demo":
        make_demo(Path(args.input_dir), Path(args.output_dir))
    elif args.cmd == "forward":
        forward(
            Path(args.input_dir),
            Path(args.output_dir),
            Path(args.manifest),
            parse_levels(args.levels),
            max(1, args.rounds),
            args.seed,
            args.frame_permute,
        )
    elif args.cmd == "reverse":
        reverse(Path(args.input_dir), Path(args.output_dir), Path(args.manifest))
    elif args.cmd == "verify":
        verify(Path(args.dir_a), Path(args.dir_b))


if __name__ == "__main__":
    main()