"""Generate saved GT/LQ pairs using DiffPIR's classical degradation.

Dependencies: numpy pillow opencv-python scipy
Default: datasets/BSDS10 -> benchmark/testsets/BSDS10/{gt,lq}.
Resize the full image to 256x256 with bicubic interpolation before degradation.
Non-square images are stretched to the target square size.
Use the saved gaussian_kernel.npy in DiffPIR pre_calculate(y, k_tensor, 4).
Here y is the saved LQ PNG read as RGB float32 in [0, 1], shaped [1,3,H,W].
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SCALE = 4
KERNEL_SIZE = 9
KERNEL_SIGMA = math.sqrt(3)
SIGMA_N = 0.05


def gaussian_kernel():
    k1d = cv2.getGaussianKernel(ksize=KERNEL_SIZE, sigma=KERNEL_SIGMA)
    return k1d @ k1d.T


def classical_degradation(x, k, sf=4):
    """Circular blur per channel, then phase-zero decimation; no noise here.

    Pass a floating-point HWC image in [0, 1] to avoid uint8 convolution
    quantization. ndimage.convolve is the current spelling of DiffPIR's
    ndimage.filters.convolve.
    """
    x = ndimage.convolve(x, np.expand_dims(k, axis=2), mode="wrap")
    st = 0
    return x[st::sf, st::sf, ...]


def generate(hr_dir, output_dir, resize_size=256, seed=0):
    if resize_size <= 0 or resize_size % SCALE:
        raise ValueError("resize-size must be a positive multiple of 4")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if not hr_dir.is_dir():
        raise NotADirectoryError(f"HR directory does not exist: {hr_dir}")
    sources = sorted(
        (p for p in hr_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: (p.name.casefold(), p.name),
    )
    if not sources:
        raise ValueError(f"No supported images found in {hr_dir}")
    if len({p.stem.casefold() for p in sources}) != len(sources):
        raise ValueError("Source filenames must have unique stems")
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"Output must be absent or empty: {output_dir}")

    # Validate all inputs before writing output.
    for path in sources:
        with Image.open(path) as image:
            image.load()

    (output_dir / "gt").mkdir(parents=True, exist_ok=True)
    (output_dir / "lq").mkdir()
    k = gaussian_kernel()
    np.save(output_dir / "gaussian_kernel.npy", k)
    rng = np.random.RandomState(seed)
    for source in sources:
        with Image.open(source) as image:
            gt = np.array(image.convert("RGB").resize(
                (resize_size, resize_size), Image.Resampling.BICUBIC
            ), dtype=np.uint8)
        x = gt.astype(np.float64) / 255.0
        y = classical_degradation(x, k, sf=SCALE)
        y = y + rng.normal(0, SIGMA_N, y.shape)
        lq = np.rint(np.clip(y, 0, 1) * 255.0).astype(np.uint8)
        Image.fromarray(gt).save(output_dir / "gt" / f"{source.stem}.png")
        Image.fromarray(lq).save(output_dir / "lq" / f"{source.stem}.png")
        print(f"{source.name}: gt {gt.shape[:2]} -> lq {lq.shape[:2]}")

    metadata = {
        "scale": SCALE,
        "kernel_size": KERNEL_SIZE,
        "kernel_sigma": KERNEL_SIGMA,
        "sigma_n": SIGMA_N,
        "seed": seed,
    }
    (output_dir / "degradation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(sources)} image pairs to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BSDS10", help="Dataset folder name")
    parser.add_argument("--resize-size", type=int, default=256,
                        help="Resize the full image to this square size using bicubic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        generate(
            Path("datasets") / args.dataset,
            Path("benchmark") / "testsets" / args.dataset,
            args.resize_size,
            args.seed,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
