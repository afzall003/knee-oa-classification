#!/usr/bin/env python
"""Grade knee osteoarthritis (Kellgren-Lawrence 0-4) from knee X-ray images.

Runs the frozen pipeline: preprocessing -> ensemble of two custom CNNs -> prior-adjusted decision,
and saves an ensemble Grad-CAM figure for each image.

Usage:
    python predict_cli.py IMAGE [IMAGE ...] [--out DIR] [--no-gradcam]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")            # hide TensorFlow info messages
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

import matplotlib
matplotlib.use("Agg")                                         # save figures without a display
import matplotlib.pyplot as plt

import gradcam as gc
import preprocessing as pp
from predict import KneeOAPipeline

GRADE_NAMES = {0: "Healthy", 1: "Doubtful", 2: "Minimal", 3: "Moderate", 4: "Severe"}


def save_figure(img, cam, grade, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Preprocessed X-ray")
    axes[1].imshow(gc.overlay(img, cam))
    axes[1].set_title("Where the model looked (Grad-CAM)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Predicted KL grade {grade}: {GRADE_NAMES[grade]}", fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Knee OA KL-grade prediction from X-ray images.")
    parser.add_argument("images", nargs="+", type=Path, help="knee X-ray image file(s)")
    parser.add_argument("--out", type=Path, default=PROJECT / "predictions",
                        help="output folder (default: ./predictions)")
    parser.add_argument("--no-gradcam", action="store_true", help="skip Grad-CAM figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    pipe = KneeOAPipeline()
    grad_models = None if args.no_gradcam else gc.build_gradcam_models(pipe.models, layers=("block4_relu2",))

    results = []
    for path in args.images:
        if not path.exists():
            print(f"[skip] file not found: {path}", file=sys.stderr)
            continue
        result = {"file": str(path), **pipe.predict_image(path)}
        grade = result["kl_grade"]
        result["grade_name"] = GRADE_NAMES[grade]
        if grad_models:
            img = pp.load_and_preprocess(path)
            cam = gc.ensemble_gradcam(grad_models, img, grade)
            fig_path = args.out / f"{path.stem}_gradcam.png"
            save_figure(img, cam, grade, fig_path)
            result["gradcam_figure"] = str(fig_path)
        results.append(result)
        print(f"{path.name}: KL grade {grade} ({GRADE_NAMES[grade]})")

    out_json = args.out / "predictions.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n{len(results)} image(s) processed. Results: {out_json}")


if __name__ == "__main__":
    main()
