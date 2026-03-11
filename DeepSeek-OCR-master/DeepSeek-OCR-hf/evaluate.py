#!/usr/bin/env python3
"""
DeepSeek-OCR Evaluation - DEBUG VERSION
Shows exactly what the model returns and captures
"""

import os
import re
import csv
import json
from io import StringIO
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoModel, AutoTokenizer
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

MODES = {
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False, "vision_tokens": 64},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False, "vision_tokens": 100},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False, "vision_tokens": 256},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False, "vision_tokens": 400},
}

INPUT_DIR = Path("./ocr_benchmark_images")
OUTPUT_DIR = Path("./ocr_results")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "evaluation_checkpoint.json"

# ============================================================================
# GROUND TRUTH
# ============================================================================

GROUND_TRUTH = """Introduction

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35] [2] [5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38] [24] [15].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states h_{t}, as a function of the previous hidden state h_{t-1} and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2] [19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section [3.2].

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4] [27] [28] [22].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17] [18] and [9].

Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [5] [2] [35]. Here, the encoder maps an input sequence of symbol representations (x_{1},...,x_{n}) to a sequence of continuous representations z = (z_{1},...,z_{n}). Given z, the decoder then generates an output sequence (y_{1},...,y_{m}) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next."""

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

def extract_from_output(full_output):
    """
    Extract OCR text from model output.
    The output format is:
    ...debug info...
    <|ref|>...<|/det|>
    Actual OCR text here
    ## Sections...
    ...
    ======================
    image size: ...
    ======================
    """
    print(f"\n  DEBUG: Full output length: {len(full_output)} chars")
    print(f"  DEBUG: First 300 chars:\n{full_output[:300]}\n")
    
    # Look for the separator line
    if '=====================' in full_output:
        parts = full_output.split('=====================')
        # The text is usually in the part before the first separator
        # or between debug info and first separator
        text_part = parts[0] if parts else full_output
        
        # Remove common debug lines
        lines = text_part.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip debug lines
            if any(skip in line for skip in [
                'BASE:', 'NO PATCHES', 'The attention layers', 
                'UserWarning', 'torch.nn', 'site-packages',
                '-> Testing', 'Processing:', 'directly resize'
            ]):
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines).strip()
        print(f"  DEBUG: Extracted {len(text)} chars of OCR text")
        return text
    else:
        # No separator found, return everything (minus obvious debug)
        lines = full_output.split('\n')
        cleaned = [l for l in lines if not any(d in l for d in ['BASE:', 'NO PATCHES', 'UserWarning'])]
        return '\n'.join(cleaned).strip()

def test_single_image(image_path, mode_name, mode_config):
    """Test a single image and show full debug output."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name}")
    print(f"Mode: {mode_name} ({mode_config['vision_tokens']} tokens)")
    print(f"{'='*60}")
    
    # Capture BOTH stdout and stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
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
                test_compress=True
            )
        except Exception as e:
            print(f"ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    stdout_text = stdout_capture.getvalue()
    stderr_text = stderr_capture.getvalue()
    
    print(f"\n1. RETURN VALUE TYPE: {type(result)}")
    print(f"   RETURN VALUE: {result}")
    
    print(f"\n2. STDOUT LENGTH: {len(stdout_text)} chars")
    if len(stdout_text) > 0:
        print(f"   STDOUT PREVIEW (last 500 chars):\n...{stdout_text[-500:]}")
    
    print(f"\n3. STDERR LENGTH: {len(stderr_text)} chars")
    
    # Try to extract OCR text
    ocr_text = extract_from_output(stdout_text)
    
    print(f"\n4. EXTRACTED OCR TEXT ({len(ocr_text)} chars):")
    print(f"   {ocr_text[:300]}...")
    
    # Show raw comparison
    print(f"\n5. COMPARISON:")
    print(f"   Ground truth (first 200 chars): {GROUND_TRUTH[:200]}...")
    
    return ocr_text

def main():
    # Test with just one image first
    test_image = INPUT_DIR / "Liberation_Serif_scale100_ar1p00_align-center_1024x978.png"
    
    if not test_image.exists():
        # Find any image
        images = list(INPUT_DIR.rglob("*.png"))
        if not images:
            print("No images found!")
            return
        test_image = images[0]
        print(f"Using test image: {test_image}")
    
    # Test each mode
    for mode_name, mode_config in MODES.items():
        ocr_text = test_single_image(test_image, mode_name, mode_config)
        
        if ocr_text and len(ocr_text) > 100:
            print(f"\n✓ SUCCESS - got {len(ocr_text)} chars")
            # Now calculate metrics
            pred = " ".join(ocr_text.split()).lower()
            gt = " ".join(GROUND_TRUTH.split()).lower()
            
            import Levenshtein
            ed = Levenshtein.distance(pred, gt)
            acc = max(0, (1 - ed/max(len(pred), len(gt))) * 100)
            comp = len(ocr_text) / mode_config['vision_tokens']
            
            print(f"  Edit Distance: {ed}")
            print(f"  Accuracy: {acc:.1f}%")
            print(f"  Compression: {comp:.1f}x")
        else:
            print(f"\n✗ FAILED - text too short or empty")
        
        input("\nPress Enter to continue to next mode...")

if __name__ == "__main__":
    main()