#!/usr/bin/env python3
"""
DeepSeek-OCR Evaluation - DEBUG VERSION
Runs inference on the first image found, saves clean OCR text as .txt
alongside the source image, then prints diagnostic info.
Compares OCR output against ground truth using edit distance.
"""

import os
import re
import sys
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoModel, AutoTokenizer
import torch

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

def find_first_image(input_dir: Path) -> Path:
    """Return the first image found (png/jpg/jpeg) or exit."""
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images = sorted(input_dir.rglob(ext))
        if images:
            return images[0]
    print(f"ERROR: No images found under {input_dir}")
    sys.exit(1)


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
    # The original regex r'{3,}' was missing the escaped \n, causing
    # re.error ("nothing to repeat") because the unescaped '{' was
    # interpreted as a repetition quantifier with nothing preceding it.
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
        # parts[1] = "BASE: ...
# NO PATCHES
# " header block
        # parts[2] = actual OCR content
        # parts[3..] = stats footer ("image size: ..." etc.)
        # Take everything between the header block and the stats footer
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
    print(f"Ground truth loaded: {len(ground_truth)} chars")

    image_path = find_first_image(INPUT_DIR)
    print(f"Selected image: {image_path}")
    print()

    for mode_name, mode_config in MODES.items():
        print(f"--- Running mode: {mode_name} ---")

        stdout_text, stderr_text, return_value = run_inference(image_path, mode_config)

        ocr_text = extract_ocr_text(stdout_text)

        txt_path = save_ocr_text(image_path, mode_name, ocr_text)
        print(f"  OCR text written to: {txt_path}")
        print(f"  OCR text length: {len(ocr_text)} chars")

        # Compute edit distance metrics
        if ocr_text:
            import Levenshtein
            pred = " ".join(ocr_text.split()).lower()
            gt = " ".join(ground_truth.split()).lower()
            ed = Levenshtein.distance(pred, gt)
            max_len = max(len(pred), len(gt))
            acc = max(0, (1 - ed / max_len) * 100) if max_len > 0 else 0.0
            comp = len(ocr_text) / mode_config["vision_tokens"]

            print(f"  Edit Distance:     {ed}")
            print(f"  Accuracy:          {acc:.1f}%")
            print(f"  Compression Ratio: {comp:.1f}x")
        else:
            print(f"  WARNING: empty OCR output")

        print()

    print("Done. Inspect .txt files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
