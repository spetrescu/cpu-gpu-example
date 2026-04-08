import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_recipe(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_forward_images(folder: str, max_images=None):
    paths = sorted([p for p in Path(folder).glob("*_forward.png") if p.is_file()])

    items = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            x = torch.from_numpy(arr).contiguous()
            name = p.stem.replace("_forward", "")
            items.append((name, x))
        except Exception as e:
            print(f"Skipping {p}: {e}")

        if max_images is not None and len(items) >= max_images:
            break

    if not items:
        raise RuntimeError("No forward images loaded")

    return items


def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        names = [name for name, _ in chunk]
        batch = torch.stack([img for _, img in chunk], dim=0).contiguous()
        yield names, batch


def save_batch_images(batch, names, out_dir, suffix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = batch.detach().cpu().numpy()
    for i, name in enumerate(names):
        Image.fromarray(arr[i].astype(np.uint8)).save(out_dir / f"{name}_{suffix}.png")


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


PERMS = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
]


def inverse_perm_indices(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


INV_PERMS = [inverse_perm_indices(p) for p in PERMS]


def build_coordinate_grids(height, width, device):
    yy = torch.arange(height, device=device, dtype=torch.int32).view(1, height, 1, 1)
    xx = torch.arange(width, device=device, dtype=torch.int32).view(1, 1, width, 1)
    cc = torch.arange(3, device=device, dtype=torch.int32).view(1, 1, 1, 3)
    return yy, xx, cc


def make_mask(yy, xx, cc, seed):
    a = (seed % 251) + 1
    b = ((seed // 251) % 253) + 1
    c = ((seed // (251 * 253)) % 127) + 1
    d = (seed * 1103515245 + 12345) & 0x7FFFFFFF

    mask = (yy * a + xx * b + cc * c + d) & 255
    return mask.to(torch.uint8)


def stage_inverse(x, yy, xx, cc, stage, substeps):
    dy = stage["dy"]
    dx = stage["dx"]
    xor_key = stage["xor_key"]
    add_key = stage["add_key"]
    inv_perm = INV_PERMS[stage["perm_id"]]

    for s in reversed(range(substeps)):
        dy_s = (dy + 17 * s) % x.shape[1]
        dx_s = (dx + 29 * s) % x.shape[2]
        xor_seed_s = xor_key + 1009 * s
        add_s = (add_key + 37 * s) % 256

        # TODO 1: undo channel permutation

        # TODO 2: undo modular addition

        # TODO 3: recreate same mask and undo XOR

        # TODO 4: undo horizontal and vertical rolls

    return x


def inverse_transform(x, recipe, device):
    _, h, w, c = x.shape
    expected = (recipe["height"], recipe["width"], recipe["channels"])
    if (h, w, c) != expected:
        raise RuntimeError(f"Batch shape {(h, w, c)} does not match recipe {expected}")

    yy, xx, cc = build_coordinate_grids(h, w, device)
    for stage in reversed(recipe["stages"]):
        x = stage_inverse(x, yy, xx, cc, stage, recipe["substeps"])
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="out_forward")
    parser.add_argument("--output_dir", type=str, default="out_reconstructed")
    parser.add_argument("--recipe", type=str, default="recipe.json")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--save_outputs", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    recipe = load_recipe(args.recipe)
    items = load_forward_images(args.input_dir, max_images=args.max_images)

    expected_shape = (recipe["height"], recipe["width"], recipe["channels"])
    for name, img in items:
        if tuple(img.shape) != expected_shape:
            raise RuntimeError(
                f"Forward image {name} has shape {tuple(img.shape)}, expected {expected_shape}"
            )

    print(f"Loaded {len(items)} forward images")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Stages: {recipe['num_stages']}")
    print(f"Substeps: {recipe['substeps']}")
    print(f"Repeat: {args.repeat}")
    print(f"Save outputs: {args.save_outputs}")

    total_compute_time = 0.0
    total_save_time = 0.0
    total_images_processed = 0

    for batch_idx, (names, batch_cpu) in enumerate(make_batches(items, args.batch_size), start=1):
        if args.device == "cuda":
            batch = batch_cpu.pin_memory().to("cuda", non_blocking=True)
        else:
            batch = batch_cpu.clone()

        sync(args.device)
        t0 = time.perf_counter()

        z = None
        for _ in range(args.repeat):
            z = inverse_transform(batch, recipe, args.device)

        sync(args.device)
        t1 = time.perf_counter()

        compute_time = t1 - t0
        total_compute_time += compute_time
        total_images_processed += batch.shape[0] * args.repeat

        print(
            f"batch {batch_idx}: {batch.shape[0]} images, "
            f"compute {compute_time:.4f} s, "
            f"{(batch.shape[0] * args.repeat) / compute_time:.2f} img/s"
        )

        if args.save_outputs:
            t_save0 = time.perf_counter()
            save_batch_images(z, names, args.output_dir, "reconstructed")
            t_save1 = time.perf_counter()
            save_time = t_save1 - t_save0
            total_save_time += save_time
            print(f"           save    {save_time:.4f} s")

    print(f"\nInverse compute total time: {total_compute_time:.4f} s")
    print(
        f"Inverse compute throughput: "
        f"{total_images_processed / total_compute_time:.2f} img/s"
    )

    if args.save_outputs:
        print(f"Output save total time: {total_save_time:.4f} s")


if __name__ == "__main__":
    main()