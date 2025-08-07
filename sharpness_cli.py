#!/usr/bin/env python3
"""
Sharpness Detection CLI

Computes a numeric sharpness score for images. Lower scores indicate blurrier images.
Default metric is Variance of Laplacian; Tenengrad is also available.

Examples:
  - Single image (human-readable):
      python sharpness_cli.py /path/to/image.jpg
  - Multiple images to JSON:
      python sharpness_cli.py /images --recursive --json
  - Use Tenengrad metric with threshold:
      python sharpness_cli.py photo.png --metric tenengrad --threshold 1.0e4
  - Demo with synthetic images:
      python sharpness_cli.py --demo

Exit codes:
  0: success
  1: one or more errors while processing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import cv2


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


@dataclass
class SharpnessResult:
    path: str
    score: float
    metric: str
    is_blurry: Optional[bool]
    error: Optional[str] = None


def load_image_grayscale(image_path: str) -> np.ndarray:
    """
    Load an image from disk, apply EXIF orientation, convert to grayscale.
    Returns an 8-bit grayscale numpy array with shape (H, W).
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img_gray = img.convert("L")
        gray = np.array(img_gray, dtype=np.uint8)
        return gray


def compute_variance_of_laplacian(gray: np.ndarray, ksize: int = 3) -> float:
    if ksize % 2 == 0 or ksize <= 0:
        raise ValueError("Laplacian ksize must be a positive odd integer")
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    # Variance of Laplacian is a widely used blur metric; lower = blurrier
    return float(lap.var())


def compute_tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    if ksize % 2 == 0 or ksize <= 0:
        raise ValueError("Sobel ksize must be a positive odd integer")
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Tenengrad: mean of squared gradient magnitude; lower = blurrier
    grad_sq = gx * gx + gy * gy
    return float(np.mean(grad_sq))


def get_metric_function(metric: str):
    metric_lower = metric.lower()
    if metric_lower in {"lap", "laplacian", "vol", "varlap"}:
        return "laplacian", compute_variance_of_laplacian
    if metric_lower in {"tenengrad", "sobel", "teng"}:
        return "tenengrad", compute_tenengrad
    raise ValueError(f"Unknown metric: {metric}")


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def iter_image_paths(paths: List[str], recursive: bool) -> Iterable[str]:
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if is_image_file(fpath):
                        yield fpath
                if not recursive:
                    break
        else:
            if is_image_file(p):
                yield p


def evaluate_image(
    image_path: str,
    metric_name: str,
    metric_fn,
    lap_ksize: int,
    sobel_ksize: int,
    threshold: Optional[float],
) -> SharpnessResult:
    try:
        gray = load_image_grayscale(image_path)
        if metric_name == "laplacian":
            score = metric_fn(gray, ksize=lap_ksize)
        else:
            score = metric_fn(gray, ksize=sobel_ksize)
        is_blurry = None if threshold is None else bool(score < threshold)
        return SharpnessResult(path=image_path, score=score, metric=metric_name, is_blurry=is_blurry)
    except Exception as exc:
        return SharpnessResult(path=image_path, score=float("nan"), metric=metric_name, is_blurry=None, error=str(exc))


def print_human(results: List[SharpnessResult], threshold: Optional[float]) -> None:
    if threshold is not None:
        print(f"Threshold: {threshold}")
    for r in results:
        if r.error:
            print(f"ERROR  | {r.path}: {r.error}")
            continue
        blur_info = ""
        if threshold is not None:
            blur_info = "  -> blurry" if r.is_blurry else "  -> sharp"
        print(f"{r.metric:<10} {r.score:>12.4f}  {r.path}{blur_info}")


def write_json(results: List[SharpnessResult], output: Optional[str]) -> None:
    payload = [
        {
            "path": r.path,
            "score": r.score,
            "metric": r.metric,
            "is_blurry": r.is_blurry,
            "error": r.error,
        }
        for r in results
    ]
    text = json.dumps(payload, indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


def write_csv(results: List[SharpnessResult], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "score", "metric", "is_blurry", "error"])
        for r in results:
            writer.writerow([r.path, r.score, r.metric, r.is_blurry, r.error])


def run_demo(metric_name: str, lap_ksize: int, sobel_ksize: int) -> None:
    size = 256
    # Create a sharp checkerboard
    tile = 16
    y, x = np.indices((size, size))
    checker = (((x // tile) + (y // tile)) % 2) * 255
    checker = checker.astype(np.uint8)

    # Create a blurred version
    blurred = cv2.GaussianBlur(checker, (9, 9), 3)

    # Compute scores
    _, metric_fn = get_metric_function(metric_name)
    sharp_score = metric_fn(checker, ksize=lap_ksize if metric_name == "laplacian" else sobel_ksize)
    blur_score = metric_fn(blurred, ksize=lap_ksize if metric_name == "laplacian" else sobel_ksize)

    print("Demo metric:", metric_name)
    print(f"Sharp synthetic score:  {sharp_score:.4f}")
    print(f"Blurred synthetic score:{blur_score:.4f}")
    print("(Lower = blurrier)")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image sharpness detector (lower score = blurrier)")
    parser.add_argument("paths", nargs="*", help="Image file(s) or directory(ies)")
    parser.add_argument("--metric", default="laplacian", help="Metric to use: laplacian (default) or tenengrad")
    parser.add_argument("--lap-ksize", type=int, default=3, help="Kernel size for Laplacian (odd integer)")
    parser.add_argument("--sobel-ksize", type=int, default=3, help="Kernel size for Sobel/Tenengrad (odd integer)")
    parser.add_argument("--threshold", type=float, default=None, help="Flag image as blurry if score < threshold")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--json", dest="json_out", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--json-file", dest="json_file", default=None, help="Write JSON to a file path")
    parser.add_argument("--csv", dest="csv_file", default=None, help="Write CSV to a file path")
    parser.add_argument("--sort", action="store_true", help="Sort results by score ascending (blurriest first)")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic demo and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        metric_name, metric_fn = get_metric_function(args.metric)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.demo:
        run_demo(metric_name, args.lap_ksize, args.sobel_ksize)
        return 0

    if not args.paths:
        print("Provide at least one image path or use --demo", file=sys.stderr)
        return 1

    to_process = list(iter_image_paths(args.paths, args.recursive))
    if not to_process:
        print("No images found to process", file=sys.stderr)
        return 1

    results: List[SharpnessResult] = []
    any_error = False

    for p in to_process:
        r = evaluate_image(
            image_path=p,
            metric_name=metric_name,
            metric_fn=metric_fn,
            lap_ksize=args.lap_ksize,
            sobel_ksize=args.sobel_ksize,
            threshold=args.threshold,
        )
        if r.error is not None:
            any_error = True
        results.append(r)

    if args.sort:
        results.sort(key=lambda r: (np.inf if np.isnan(r.score) else r.score))

    wrote_any = False

    if args.json_out:
        write_json(results, None)
        wrote_any = True

    if args.json_file:
        write_json(results, args.json_file)
        wrote_any = True

    if args.csv_file:
        write_csv(results, args.csv_file)
        wrote_any = True

    if not wrote_any:
        print_human(results, args.threshold)

    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())