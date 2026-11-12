"""Run UPSR-DC inference and benchmark evaluation from Gradio."""

import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import gradio as gr
import yaml


ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "benchmark"
TESTSETS = BENCHMARK / "testsets"
RESULTS = BENCHMARK / "results"
METRICS = BENCHMARK / "metrics"
CONFIG = ROOT / "options" / "003_UPSR_DC_RealSR_x4.yml"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PARAMETERS = {
    "noise_alpha": (0.4, ("dc", "noise_alpha")),
    "min_noise_level": (0.1, ("diffusion", "params", "min_noise_level")),
    "min_noise": (0.3, ("diffusion", "min_noise")),
    "rho": (0.4, ("dc", "rho")),
    "kappa": (2.0, ("diffusion", "params", "kappa")),
}


def contains_images(paths):
    return any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in paths)


def available_datasets(for_evaluation=False):
    datasets = []
    for folder in sorted(TESTSETS.glob("*")):
        if not contains_images((folder / "lq").glob("*")):
            continue
        if for_evaluation:
            results = RESULTS / folder.name
            if contains_images((folder / "gt").glob("*")) and contains_images(
                path for path in results.rglob("*") if path.parent != results
            ):
                datasets.append(folder.name)
        elif (folder / "gaussian_kernel.npy").is_file():
            datasets.append(folder.name)
    return datasets


def refresh_datasets(selected, for_evaluation=False):
    datasets = available_datasets(for_evaluation)
    selected = selected if selected in datasets else next(iter(datasets), None)
    return gr.Dropdown(choices=datasets, value=selected, label="Dataset")


def run_script(label, script, *arguments, env=None):
    result = subprocess.run(
        [sys.executable, "-u", script, *arguments],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        detail = "\n".join(result.stdout.splitlines()[-20:])
        raise gr.Error(f"{label} thất bại ({result.returncode}).\n{detail}")


def run_inference(dataset, *values):
    if dataset not in available_datasets():
        raise gr.Error("Chọn dataset có ảnh trong lq/ và file gaussian_kernel.npy.")

    effective_values = {name: default for name, (default, _) in PARAMETERS.items()}
    for name, raw_value in zip(PARAMETERS, values):
        if raw_value is None or not raw_value.strip():
            continue
        try:
            value = float(raw_value)
        except ValueError:
            raise gr.Error(f"{name} phải là số.") from None
        if not math.isfinite(value) or value < 0:
            raise gr.Error(f"{name} phải là số hữu hạn và không âm.")
        if name in ("min_noise_level", "rho", "kappa") and value == 0:
            raise gr.Error(f"{name} phải lớn hơn 0.")
        if name == "min_noise" and value > 1:
            raise gr.Error("min_noise phải nằm trong [0, 1].")
        effective_values[name] = value

    method = "upsr_dc" + "".join(f"_{name}_{value}" for name, value in effective_values.items())
    destination = RESULTS / dataset / method
    if destination.exists():
        raise gr.Error(f"Method đã có thư mục kết quả: {destination.relative_to(ROOT)}")

    data_dir = TESTSETS / dataset
    experiment = f"gradio_{uuid4().hex[:12]}"
    with CONFIG.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["name"] = experiment
    config["dc"].update(enabled=True, kernel_path=str(data_dir / "gaussian_kernel.npy"))
    config["datasets"]["val_1"].update(name=dataset, dataroot_lq=str(data_dir / "lq"))
    for name, (_, config_path_parts) in PARAMETERS.items():
        *sections, key = config_path_parts
        section = config
        for part in sections:
            section = section[part]
        section[key] = effective_values[name]

    with TemporaryDirectory(prefix=".upsr_gradio_", dir=BENCHMARK) as temp:
        temp_dir = Path(temp)
        config_path = temp_dir / "run.yml"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        run_script("Inference", "test.py", "-opt", str(config_path))

        source = ROOT / "results" / experiment / "visualization" / dataset
        images = sorted(source.glob("*.png"))
        if not images:
            raise gr.Error(f"Không tìm thấy ảnh SR trong {source.relative_to(ROOT)}")

        staged = temp_dir / "images"
        staged.mkdir()
        for image in images:
            shutil.copy2(image, staged / image.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(destination)

    gallery = [(str(destination / image.name), image.name) for image in images]
    return method, f"Đã lưu {len(images)} ảnh vào {destination.relative_to(ROOT).as_posix()}", gallery


def run_evaluation(dataset, crop_border=""):
    if dataset not in available_datasets(for_evaluation=True):
        raise gr.Error("Chọn dataset có ảnh LR, GT và ít nhất một thư mục kết quả SR.")

    try:
        border = int((crop_border or "").strip() or "4")
    except ValueError:
        raise gr.Error("Crop border phải là số nguyên không âm.") from None
    if border < 0:
        raise gr.Error("Crop border phải là số nguyên không âm.")
    run_script(
        "Evaluate", "evaluate.py", f"--dataset={dataset}", f"--crop-border={border}",
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    destination = METRICS / dataset
    summary = destination / "summary.csv"
    per_image = destination / "per_image.csv"
    if not summary.is_file() or not per_image.is_file():
        raise gr.Error(f"Không tìm thấy CSV trong {destination.relative_to(ROOT)}")
    with summary.open(encoding="utf-8", newline="") as file:
        headers, *rows = csv.reader(file)
    table = {"headers": headers, "data": rows}
    gallery = [(str(path), path.stem) for path in sorted((destination / "visualizations").glob("*.png"))]
    return (
        f"Đã đánh giá {len(rows)} method. Kết quả tại {destination.relative_to(ROOT).as_posix()}",
        table,
        [str(summary), str(per_image)],
        gallery,
    )


def build_app():
    with gr.Blocks(title="UPSR-DC", analytics_enabled=False) as demo:
        gr.Markdown("# UPSR-DC")
        with gr.Tab("Inference"):
            with gr.Row():
                dataset = refresh_datasets(None)
                refresh = gr.Button("Làm mới dataset")
            gr.Markdown("Ô bỏ trống dùng giá trị gợi ý. Tên method ghi các giá trị thực tế của lần chạy.")
            with gr.Row():
                inputs = [
                    gr.Textbox(label=name, value="", placeholder=str(default))
                    for name, (default, _) in PARAMETERS.items()
                ]
            run = gr.Button("Chạy inference", variant="primary")
            method = gr.Textbox(label="Method", interactive=False)
            status = gr.Textbox(label="Kết quả", interactive=False)
            gallery = gr.Gallery(label="Ảnh SR", columns=4)

            refresh.click(refresh_datasets, inputs=dataset, outputs=dataset, queue=False)
            run.click(run_inference, inputs=[dataset, *inputs], outputs=[method, status, gallery],
                      concurrency_id="benchmark")

        with gr.Tab("Evaluate") as evaluate_tab:
            with gr.Row():
                evaluation_dataset = refresh_datasets(None, for_evaluation=True)
                crop_border = gr.Textbox(label="Crop border (px)", value="", placeholder="4")
                evaluation_refresh = gr.Button("Làm mới dataset")
            evaluate = gr.Button("Chạy evaluate", variant="primary")
            evaluation_status = gr.Textbox(label="Kết quả", interactive=False)
            summary_table = gr.Dataframe(label="Điểm trung bình theo method", interactive=False)
            csv_files = gr.File(label="Tải CSV", file_count="multiple", interactive=False)
            comparisons = gr.Gallery(label="So sánh LR, GT và SR", columns=2)

            def refresh_evaluation_datasets(selected):
                return refresh_datasets(selected, for_evaluation=True)

            evaluation_refresh.click(refresh_evaluation_datasets, evaluation_dataset, evaluation_dataset, queue=False)
            evaluate_tab.select(refresh_evaluation_datasets, evaluation_dataset, evaluation_dataset, queue=False)
            evaluate.click(run_evaluation, inputs=[evaluation_dataset, crop_border],
                           outputs=[evaluation_status, summary_table, csv_files, comparisons],
                           concurrency_id="benchmark")
    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(allowed_paths=[str(RESULTS), str(METRICS)])
