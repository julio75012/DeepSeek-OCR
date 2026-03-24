#!/usr/bin/env python3
"""
DeepSeek-OCR Evaluation
Runs inference on all images found in INPUT_DIR across all configured modes,
saves clean OCR text as .txt alongside the source image, computes edit-distance
accuracy and compression ratio, and persists results to a CSV file.

Supports resumption: previously computed (image, mode) pairs found in the CSV
are skipped automatically.
"""

import csv
import os
import re
import sys
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoModel, AutoTokenizer
import torch
import Levenshtein

# ============================================================================
# CONFIGURATION
# ============================================================================

MODES = {
    "Tiny":  {"base_size": 512,  "image_size": 512,  "crop_mode": False, "vision_tokens": 64},
    "Small": {"base_size": 640,  "image_size": 640,  "crop_mode": False, "vision_tokens": 100},
    "Base":  {"base_size": 1024, "image_size": 1024, "crop_mode": False, "vision_tokens": 256},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False, "vision_tokens": 400},
}

INPUT_DIR = Path("./ocr_benchmark_images")
OUTPUT_DIR = Path("./ocr_benchmark_text")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "results.csv"
CSV_COLUMNS = ["image", "mode", "accuracy", "compression_ratio", "edit_distance", "ocr_length"]

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.txt"

# ============================================================================
# GROUND TRUTH LOADING
# ============================================================================

def load_ground_truth() -> str:
    """Load ground truth from external text file."""
    if not GROUND_TRUTH_FILE.exists():
        print(f"ERROR: Ground truth file not found: {GROUND_TRUTH_FILE}")
        sys.exit(1)
    return GROUND_TRUTH_FILE.read_text(encoding="utf-8").strip()

# ============================================================================
# CSV RESUME HELPERS
# ============================================================================

def load_completed_keys(csv_path: Path) -> set[tuple[str, str]]:
    """
    Load already-computed (image, mode) pairs from the CSV file.
    Returns a set of (image_filename, mode_name) tuples.
    """
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["image"], row["mode"]))
    return completed


def append_result_row(csv_path: Path, row: dict) -> None:
    """
    Append a single result row to the CSV file.
    Creates the file with headers if it does not yet exist.
    """
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ============================================================================
# MODEL LOADING
# ============================================================================

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-ocr", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "deepseek-ai/deepseek-ocr",
    _attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
).eval().to(device)
if torch.cuda.is_available():
    model = model.to(torch.bfloat16)
print(f"Model loaded on {device}")

# ============================================================================
# HELPERS
# ============================================================================

def find_all_images(input_dir: Path) -> list[Path]:
    """Return all images found (png/jpg/jpeg), sorted for determinism."""
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(input_dir.rglob(ext))
    images = sorted(set(images))
    if not images:
        print(f"ERROR: No images found under {input_dir}")
        sys.exit(1)
    return images


def run_inference(image_path: Path, mode_config: dict) -> tuple[str, str, object]:
    """
    Run model.infer(), capturing stdout/stderr separately.
    Returns (stdout_text, stderr_text, return_value).
    """
    stdout_buf = StringIO()
    stderr_buf = StringIO()
    result = None

    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        try:
            result = model.infer(
                tokenizer,
                prompt="<image>\n<|grounding|>Convert the document to markdown.",
                image_file=str(image_path),
                output_path="./temp_output",
                base_size=mode_config["base_size"],
                image_size=mode_config["image_size"],
                crop_mode=mode_config["crop_mode"],
                save_results=False,
                test_compress=True,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()

    return stdout_buf.getvalue(), stderr_buf.getvalue(), result


def strip_grounding_tags(text: str) -> str:
    """
    Remove all DeepSeek-OCR grounding/detection markup from text.

    Strips patterns like:
      <|ref|>sub_title<|/ref|><|det|>[[420, 10, 576, 22]]<|/det|>
      <|ref|>text<|/ref|><|det|>[[80, 277, 916, 435]]<|/det|>

    Preserves all other content including markdown headings (##).
    """
    # Remove <|ref|>...<|/ref|> tags and their content
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
    # Remove <|det|>...<|/det|> tags and their content
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    # Remove any other remaining special tokens of the form <|...|>
    text = re.sub(r'<\|[^|]*\|>', '', text)
    # Collapse 3+ consecutive newlines into exactly 2 newlines.
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_ocr_text(stdout_text: str) -> str:
    """
    Extract and clean OCR text from captured stdout.

    The model prints output in this format:
        =====================
        BASE: ...
        NO PATCHES
        =====================
        <|ref|>...<|/ref|><|det|>...<|/det|>
        ## Heading
        Actual OCR text...
        ==================================================
        image size: ...
        ==================================================

    We extract the content between the first and last separator blocks,
    then strip grounding tags.
    """
    # Split on the separator lines
    sep_pattern = r'={5,}'
    parts = re.split(sep_pattern, stdout_text)

    if len(parts) >= 3:
        # parts[0] = before first separator (empty or preamble)
        # parts[1] = "BASE: ... / NO PATCHES / " header block
        # parts[2] = actual OCR content
        # parts[3..] = stats footer ("image size: ..." etc.)
        ocr_body = parts[2] if len(parts) > 2 else ""
    else:
        # Fallback: use entire stdout
        ocr_body = stdout_text

    return strip_grounding_tags(ocr_body)


def save_ocr_text(image_path: Path, mode_name: str, ocr_text: str) -> Path:
    """
    Write clean OCR text to a .txt file in OUTPUT_DIR.
    Naming: <image_stem>__<mode>.txt
    """
    txt_name = f"{image_path.stem}__{mode_name}.txt"
    txt_path = OUTPUT_DIR / txt_name
    txt_path.write_text(ocr_text, encoding="utf-8")
    return txt_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    ground_truth = load_ground_truth()
    gt_normalized = " ".join(ground_truth.split()).lower()
    print(f"Ground truth loaded: {len(ground_truth)} chars")

    images = find_all_images(INPUT_DIR)
    print(f"Found {len(images)} image(s) in {INPUT_DIR}")

    # Load already-computed results to enable resumption
    completed = load_completed_keys(RESULTS_CSV)
    if completed:
        print(f"Resuming: {len(completed)} (image, mode) pair(s) already computed -- will be skipped.")
    print()

    total_pairs = len(images) * len(MODES)
    done_count = len(completed)

    for img_idx, image_path in enumerate(images, start=1):
        image_name = image_path.name
        print(f"[Image {img_idx}/{len(images)}] {image_name}")

        for mode_name, mode_config in MODES.items():
            key = (image_name, mode_name)

            if key in completed:
                done_count_display = done_count  # already counted
                print(f"  {mode_name:>6s}: SKIPPED (already computed)")
                continue

            stdout_text, stderr_text, return_value = run_inference(image_path, mode_config)
            ocr_text = extract_ocr_text(stdout_text)
            save_ocr_text(image_path, mode_name, ocr_text)

            # Compute metrics
            if ocr_text:
                pred_normalized = " ".join(ocr_text.split()).lower()
                ed = Levenshtein.distance(pred_normalized, gt_normalized)
                max_len = max(len(pred_normalized), len(gt_normalized))
                acc = max(0.0, (1 - ed / max_len) * 100) if max_len > 0 else 0.0
                comp = len(ocr_text) / mode_config["vision_tokens"]
            else:
                ed = len(gt_normalized)
                acc = 0.0
                comp = 0.0

            # Persist result row immediately (crash-safe resumption)
            row = {
                "image": image_name,
                "mode": mode_name,
                "accuracy": f"{acc:.2f}",
                "compression_ratio": f"{comp:.2f}",
                "edit_distance": ed,
                "ocr_length": len(ocr_text),
            }
            append_result_row(RESULTS_CSV, row)
            completed.add(key)
            done_count += 1

            print(f"  {mode_name:>6s}: Acc={acc:5.1f}%  ED={ed:5d}  Comp={comp:5.1f}x  Len={len(ocr_text)}  [{done_count}/{total_pairs}]")

        print()

    print(f"All done. Results saved to: {RESULTS_CSV}")
    print(f"OCR text files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
