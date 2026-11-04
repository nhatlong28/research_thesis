"""Evaluate SR folders with PSNR, SSIM, MS-SSIM, LPIPS-VGG and TorchMetrics/PyIQA CLIP-IQA."""

import argparse
import csv
from pathlib import Path
from statistics import fmean

import numpy as np
import matplotlib.pyplot as plt
import pyiqa
from PIL import Image
import torch
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal import CLIPImageQualityAssessment


SCORES = ("psnr_rgb", "ssim_rgb", "msssim_rgb", "lpips_vgg", "clip_iqa", "clipiqa")
MS_SSIM_MIN_SIZE = 176  # TorchMetrics default: five scales and an 11-pixel kernel.


def read_rgb(path, device):
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"))
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32
    ) / 255.0


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def visualize(comparisons, lq_dir, gt_dir, output_dir, methods):
    """Save one comparison figure per image: LR, GT, then each method's SR."""
    lq_files = {p.stem: p for p in lq_dir.iterdir() if p.is_file()}
    gt_files = {p.stem: p for p in gt_dir.iterdir() if p.is_file()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, sr_paths in comparisons.items():
        fig, axes = plt.subplots(1, len(methods) + 2,
                                 figsize=(4 * (len(methods) + 2), 4), squeeze=False)
        try:
            panels = [("LR (LQ)", lq_files[name]), ("HR (GT)", gt_files[name])]
            panels += [(f"{method} (SR)", sr_paths.get(method)) for method in methods]
            for ax, (label, path) in zip(axes[0], panels):
                ax.axis("off")
                if path is None:
                    ax.set_title(f"{label}\nNo result")
                    continue
                with Image.open(path) as image:
                    ax.imshow(image.convert("RGB"), interpolation="nearest")
                    ax.set_title(f"{label}\n{image.width} x {image.height}")
            fig.suptitle(name)
            fig.tight_layout()
            fig.savefig(output_dir / f"{name}.png", dpi=150)
        finally:
            plt.close(fig)
    print(f"Comparison figures saved to {output_dir}")


def evaluate(gt_dir, results_dir, output_dir, border=4, device="auto", lq_dir=None):
    if border < 0:
        raise ValueError("crop-border must be nonnegative")
    gt_files = sorted(p for p in gt_dir.iterdir() if p.is_file())

    # Each image folder is one method/run; keep nested sweep runs separate.
    runs = {}
    for folder in sorted(results_dir.rglob("*")):
        if folder.is_dir():
            files = sorted(p for p in folder.iterdir() if p.is_file())
            if files:
                runs[folder.relative_to(results_dir).as_posix()] = files
            else:
                print(f"Skip {folder.name}: empty")
    if not runs:
        print("No SR images found.")
        return

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}; reference crop: {border}; RGB [0, 1]")
    metrics = {
        "psnr_rgb": PeakSignalNoiseRatio(data_range=1.0),
        "ssim_rgb": StructuralSimilarityIndexMeasure(data_range=1.0),
        "msssim_rgb": MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0),
        "lpips_vgg": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True),
        "clip_iqa": CLIPImageQualityAssessment(data_range=1.0, prompts=("quality",)),
        "clipiqa": pyiqa.create_metric("clipiqa", device=device),
    }
    for metric in metrics.values():
        metric.to(device).eval()

    per_image, summary = [], []
    comparisons = {}
    with torch.inference_mode():
        for method, files in runs.items():
            rows, used_gt = [], set()
            for sr_path in files:
                # Prefer exact names; otherwise match a GT stem inside the SR stem.
                matches = [p for p in gt_files if p.stem == sr_path.stem]
                if not matches:
                    matches = [p for p in gt_files if p.stem in sr_path.stem]
                if len(matches) != 1:
                    raise ValueError(f"{method}/{sr_path.name}: expected one matching GT")
                gt_path = matches[0]
                if gt_path in used_gt:
                    raise ValueError(f"{method}: multiple SR files match {gt_path.name}")
                used_gt.add(gt_path)
                comparisons.setdefault(gt_path.stem, {})[method] = sr_path

                sr, gt = read_rgb(sr_path, device), read_rgb(gt_path, device)
                if sr.shape != gt.shape:
                    raise ValueError(f"{method}/{sr_path.name}: SR and GT sizes differ")
                if min(sr.shape[-2:]) - 2 * border < 32:
                    raise ValueError(f"{method}/{sr_path.name}: cropped image too small")
                sr_ref = sr[..., border:-border, border:-border] if border else sr
                gt_ref = gt[..., border:-border, border:-border] if border else gt

                row = {"method": method, "image": sr_path.name}
                for name, metric in metrics.items():
                    if name == "msssim_rgb" and min(sr_ref.shape[-2:]) < MS_SSIM_MIN_SIZE:
                        row[name] = float("nan")
                        continue
                    if name != "clipiqa":
                        metric.reset()  # Independent TorchMetrics scores for each image.
                    score = metric(sr) if name in ("clip_iqa", "clipiqa") else metric(sr_ref, gt_ref)
                    row[name] = score.item()
                rows.append(row)
                print(f"{method}/{sr_path.name}: " + ", ".join(
                    f"{name}={row[name]:.6f}" for name in SCORES
                ), flush=True)

            per_image.extend(rows)
            summary.append({
                "method": method,
                "num_images": len(rows),
                **{name: fmean(row[name] for row in rows) for name in SCORES},
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_image.csv", per_image)
    write_csv(output_dir / "summary.csv", summary)
    visualize(comparisons, lq_dir or gt_dir.parent / "lq", gt_dir,
              output_dir / "visualizations", runs)
    print("\nMethod | N | PSNR up | SSIM up | MS-SSIM up | LPIPS down | "
          "CLIP-IQA (TorchMetrics) up | CLIPIQA (PyIQA) up")
    for row in summary:
        print(f"{row['method']} | {row['num_images']} | " + " | ".join(
            f"{row[name]:.6f}" for name in SCORES
        ))
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=Path("benchmark/testsets/BSDS10/gt"))
    parser.add_argument("--lq-dir", type=Path, help="LR folder; default: sibling lq of gt-dir")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/metrics"))
    parser.add_argument("--crop-border", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args.gt_dir, args.results_dir, args.output_dir,
             args.crop_border, args.device, args.lq_dir)


if __name__ == "__main__":
    main()
