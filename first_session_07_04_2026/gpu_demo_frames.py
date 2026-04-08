
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


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


def _rng_for(base_seed: int, level_idx: int, round_idx: int, stream_id: int, frame_idx: int = 0):
    seed = (
        int(base_seed) * 0x9E3779B1
        + level_idx * 0x85EBCA6B
        + round_idx * 0xC2B2AE35
        + int(stream_id) * 0x27D4EB2F
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


def frames_to_tensor(frames: List[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(arr).to(device=device, dtype=torch.uint8)


def tensor_to_frames(x: torch.Tensor) -> List[np.ndarray]:
    arr = x.detach().cpu().numpy()
    return [arr[i] for i in range(arr.shape[0])]


def _tiles_view(x: torch.Tensor, tile: int) -> torch.Tensor:
    b, h, w, c = x.shape
    rows, cols = h // tile, w // tile
    return x.view(b, rows, tile, cols, tile, c).permute(0, 1, 3, 2, 4, 5).contiguous()


def _apply_tile_permutation_forward(x: torch.Tensor, tile: int, perm_tiles: torch.Tensor) -> torch.Tensor:
    b, h, w, c = x.shape
    rows, cols = h // tile, w // tile
    total = rows * cols
    tiles = _tiles_view(x, tile).view(b, total, tile, tile, c)
    idx = perm_tiles[:, :, None, None, None].expand(b, total, tile, tile, c)
    out = torch.gather(tiles, 1, idx)
    out = out.view(b, rows, cols, tile, tile, c).permute(0, 1, 3, 2, 4, 5).contiguous()
    return out.view(b, h, w, c)


def _apply_tile_permutation_inverse(x: torch.Tensor, tile: int, inv_tiles: torch.Tensor) -> torch.Tensor:
    b, h, w, c = x.shape
    rows, cols = h // tile, w // tile
    total = rows * cols
    tiles = _tiles_view(x, tile).view(b, total, tile, tile, c)
    idx = inv_tiles[:, :, None, None, None].expand(b, total, tile, tile, c)
    out = torch.gather(tiles, 1, idx)
    out = out.view(b, rows, cols, tile, tile, c).permute(0, 1, 3, 2, 4, 5).contiguous()
    return out.view(b, h, w, c)


def _apply_in_tile_perm_forward(x: torch.Tensor, tile: int, perm_pix: torch.Tensor) -> torch.Tensor:
    b, h, w, c = x.shape
    rows, cols = h // tile, w // tile
    total = rows * cols
    tiles = _tiles_view(x, tile).view(b, total, tile * tile, c)
    idx = perm_pix[:, None, :, None].expand(b, total, tile * tile, c)
    out = torch.gather(tiles, 2, idx)
    out = out.view(b, rows, cols, tile, tile, c).permute(0, 1, 3, 2, 4, 5).contiguous()
    return out.view(b, h, w, c)


def _apply_in_tile_perm_inverse(x: torch.Tensor, tile: int, inv_pix: torch.Tensor) -> torch.Tensor:
    b, h, w, c = x.shape
    rows, cols = h // tile, w // tile
    total = rows * cols
    tiles = _tiles_view(x, tile).view(b, total, tile * tile, c)
    idx = inv_pix[:, None, :, None].expand(b, total, tile * tile, c)
    out = torch.gather(tiles, 2, idx)
    out = out.view(b, rows, cols, tile, tile, c).permute(0, 1, 3, 2, 4, 5).contiguous()
    return out.view(b, h, w, c)


def build_perm_batch(batch_frame_indices: List[int], total_tiles: int, tile: int, seed: int, li: int, ri: int, device: torch.device):
    perms_tiles = []
    invs_tiles = []
    perms_pix = []
    invs_pix = []
    for fi in batch_frame_indices:
        rng_tiles = _rng_for(seed, li, ri, 101, fi)
        perm_tiles, inv_tiles = _make_perm(total_tiles, rng_tiles)
        rng_pix = _rng_for(seed, li, ri, 202, fi)
        perm_pix, inv_pix = _make_perm(tile * tile, rng_pix)
        perms_tiles.append(perm_tiles)
        invs_tiles.append(inv_tiles)
        perms_pix.append(perm_pix)
        invs_pix.append(inv_pix)
    return (
        torch.from_numpy(np.stack(perms_tiles)).to(device=device, dtype=torch.long),
        torch.from_numpy(np.stack(invs_tiles)).to(device=device, dtype=torch.long),
        torch.from_numpy(np.stack(perms_pix)).to(device=device, dtype=torch.long),
        torch.from_numpy(np.stack(invs_pix)).to(device=device, dtype=torch.long),
    )


@torch.no_grad()
def transform_forward_batch(x: torch.Tensor, batch_frame_indices: List[int], levels: List[int], rounds: int, seed: int) -> torch.Tensor:
    cur = x
    for li, tile in enumerate(levels):
        total_tiles = (cur.shape[1] // tile) * (cur.shape[2] // tile)
        for ri in range(rounds):
            perm_tiles, _, perm_pix, _ = build_perm_batch(batch_frame_indices, total_tiles, tile, seed, li, ri, cur.device)
            cur = _apply_tile_permutation_forward(cur, tile, perm_tiles)
            cur = _apply_in_tile_perm_forward(cur, tile, perm_pix)
    return cur


@torch.no_grad()
def transform_reverse_batch(x: torch.Tensor, batch_frame_indices: List[int], levels: List[int], rounds: int, seed: int) -> torch.Tensor:
    cur = x
    for li_rev, tile in enumerate(reversed(levels)):
        li = len(levels) - 1 - li_rev
        total_tiles = (cur.shape[1] // tile) * (cur.shape[2] // tile)
        for ri_rev in range(rounds):
            ri = rounds - 1 - ri_rev
            _, inv_tiles, _, inv_pix = build_perm_batch(batch_frame_indices, total_tiles, tile, seed, li, ri, cur.device)
            cur = _apply_in_tile_perm_inverse(cur, tile, inv_pix)
            cur = _apply_tile_permutation_inverse(cur, tile, inv_tiles)
    return cur


def build_frame_permutation(num_frames: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState((seed * 0xA24BAED3 + 0x9FB21C65) & 0xFFFFFFFF)
    return _make_perm(num_frames, rng)


def chunked(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def forward(input_dir: Path, output_dir: Path, manifest_path: Path, levels: List[int], rounds: int, seed: int, frame_permute: bool, batch_size: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")
    device = torch.device("cuda")

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
    out = [None] * len(padded)
    for batch in chunked(list(range(len(padded))), batch_size):
        x = frames_to_tensor([padded[i] for i in batch], device)
        y = transform_forward_batch(x, batch, levels, rounds, seed)
        y_frames = tensor_to_frames(y)
        for local_i, global_i in enumerate(batch):
            out[global_i] = y_frames[local_i]
        print(f"[GPU] forward batch {batch[0]}..{batch[-1]}")
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
        "gpu_forward_seconds": transform_seconds,
        "device": str(device),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {output_dir}")
    print(f"Wrote {manifest_path}")
    print(f"GPU forward time: {transform_seconds:.2f}s")


def reverse(input_dir: Path, output_dir: Path, manifest_path: Path, batch_size: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")
    device = torch.device("cuda")

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
    recovered = [None] * len(frames)
    for batch in chunked(list(range(len(frames))), batch_size):
        x = frames_to_tensor([frames[i] for i in batch], device)
        y = transform_reverse_batch(x, batch, levels, rounds, seed)
        y_frames = tensor_to_frames(y)
        for local_i, global_i in enumerate(batch):
            recovered[global_i] = unpad_frame(y_frames[local_i], int(manifest["orig_h"]), int(manifest["orig_w"]))
        print(f"[GPU] reverse batch {batch[0]}..{batch[-1]}")
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
    print(f"Wrote {output_dir}")
    print(f"GPU reverse time: {transform_seconds:.2f}s")


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

    s1 = sub.add_parser("forward")
    s1.add_argument("input_dir")
    s1.add_argument("output_dir")
    s1.add_argument("--manifest", required=True)
    s1.add_argument("--levels", default="64,32,16")
    s1.add_argument("--rounds", type=int, default=256)
    s1.add_argument("--seed", type=int, default=1337)
    s1.add_argument("--frame-permute", action="store_true")
    s1.add_argument("--batch-size", type=int, default=3)

    s2 = sub.add_parser("reverse")
    s2.add_argument("input_dir")
    s2.add_argument("output_dir")
    s2.add_argument("--manifest", required=True)
    s2.add_argument("--batch-size", type=int, default=3)

    s3 = sub.add_parser("verify")
    s3.add_argument("dir_a")
    s3.add_argument("dir_b")

    args = ap.parse_args()

    if args.cmd == "forward":
        forward(Path(args.input_dir), Path(args.output_dir), Path(args.manifest),
                parse_levels(args.levels), max(1, args.rounds), args.seed,
                args.frame_permute, max(1, args.batch_size))
    elif args.cmd == "reverse":
        reverse(Path(args.input_dir), Path(args.output_dir), Path(args.manifest),
                max(1, args.batch_size))
    elif args.cmd == "verify":
        verify(Path(args.dir_a), Path(args.dir_b))


if __name__ == "__main__":
    main()
