import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# torch numpy Pillow

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_recipe(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_images_from_dir(folder: str, max_images=None):
    paths = sorted(
        [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    )

    items = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            x = torch.from_numpy(arr).contiguous()  # H,W,C
            items.append((p.stem, x))
        except Exception as e:
            print(f"Skipping {p}: {e}")

        if max_images is not None and len(items) >= max_images:
            break

    if not items:
        raise RuntimeError("No images loaded")

    return items


def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        names = [name for name, _ in chunk]
        batch = torch.stack([img for _, img in chunk], dim=0).contiguous()  # N,H,W,C
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


def stage_forward(x, yy, xx, cc, stage, substeps):
    dy = stage["dy"]
    dx = stage["dx"]
    xor_key = stage["xor_key"]
    add_key = stage["add_key"]
    perm = PERMS[stage["perm_id"]]

    for s in range(substeps):
        dy_s = (dy + 17 * s) % x.shape[1]
        dx_s = (dx + 29 * s) % x.shape[2]
        xor_seed_s = xor_key + 1009 * s
        add_s = (add_key + 37 * s) % 256

        x = torch.roll(x, shifts=dy_s, dims=1)
        x = torch.roll(x, shifts=dx_s, dims=2)

        mask = make_mask(yy, xx, cc, xor_seed_s)
        x = torch.bitwise_xor(x, mask)

        x = ((x.to(torch.int16) + add_s) & 255).to(torch.uint8)
        x = x[..., perm]

    return x


def forward_transform(x, recipe, device):
    _, h, w, c = x.shape
    expected = (recipe["height"], recipe["width"], recipe["channels"])
    if (h, w, c) != expected:
        raise RuntimeError(f"Batch shape {(h, w, c)} does not match recipe {expected}")

    yy, xx, cc = build_coordinate_grids(h, w, device)
    for stage in recipe["stages"]:
        x = stage_forward(x, yy, xx, cc, stage, recipe["substeps"])
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="frames")
    parser.add_argument("--output_dir", type=str, default="out_forward")
    parser.add_argument("--recipe", type=str, default="recipe.json")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    recipe = load_recipe(args.recipe)
    items = load_images_from_dir(args.input_dir, max_images=args.max_images)

    expected_shape = (recipe["height"], recipe["width"], recipe["channels"])
    for name, img in items:
        if tuple(img.shape) != expected_shape:
            raise RuntimeError(
                f"Image {name} has shape {tuple(img.shape)}, expected {expected_shape}"
            )

    print(f"Loaded {len(items)} images")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Stages: {recipe['num_stages']}")
    print(f"Substeps: {recipe['substeps']}")

    total_time = 0.0
    total_images = 0

    for batch_idx, (names, batch_cpu) in enumerate(make_batches(items, args.batch_size), start=1):
        if args.device == "cuda":
            batch = batch_cpu.pin_memory().to("cuda", non_blocking=True)
        else:
            batch = batch_cpu.clone()

        sync(args.device)
        t0 = time.perf_counter()
        y = forward_transform(batch, recipe, args.device)
        sync(args.device)
        t1 = time.perf_counter()

        save_batch_images(y, names, args.output_dir, "forward")

        batch_time = t1 - t0
        total_time += batch_time
        total_images += batch.shape[0]

        print(
            f"batch {batch_idx}: {batch.shape[0]} images, "
            f"{batch_time:.4f} s, "
            f"{batch.shape[0] / batch_time:.2f} img/s"
        )

    print(f"\nForward total time: {total_time:.4f} s")
    print(f"Forward throughput: {total_images / total_time:.2f} img/s")


if __name__ == "__main__":
    main()