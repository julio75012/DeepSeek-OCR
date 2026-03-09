#!/usr/bin/env python3
"""
OCR Benchmark Image Generator
Generates 2,250 images (25 fonts × 10 scales × 9 aspect ratios)
from a reference text for testing DeepSeek-OCR accuracy.
"""

import os
import math
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np

# ============================================================================
# 1. SAMPLE TEXT - Paste your reference OCR text here
# ============================================================================
SAMPLE_TEXT = """## 1 Introduction  

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35] [2] [5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38] [24] [15].  

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_{t}$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.  

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2] [19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.  

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.  

## 2 Background  

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section [3.2].  

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4] [27] [28] [22].  

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].  

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17] [18] and [9].  

## 3 Model Architecture  

Most competitive neural sequence transduction models have an encoder-decoder structure [5] [2] [35]. Here, the encoder maps an input sequence of symbol representations $(x_{1},\\ldots,x_{n})$ to a sequence of continuous representations $z = (z_{1},\\ldots,z_{n})$. Given $z$, the decoder then generates an output sequence $(y_{1},\\ldots,y_{m})$ of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next."""

# ============================================================================
# 2. FONT CONFIGURATION - 25 Common Latin fonts
# ============================================================================
# Note: The script will attempt to find these fonts on your system.
# For best results, install Google Fonts or specify full paths.
FONTS = [
    # === STANDARD OFFICE FONTS (Times/Arial/Courier clones) ===
    "Liberation Serif",        # Times New Roman metrics - most common printed document font
    "Liberation Sans",         # Arial metrics - standard UI font
    "Liberation Mono",         # Courier metrics - code/monospace standard
    
    # === SCREEN-OPTIMIZED FONTS (Excellent hinting) ===
    "DejaVu Serif",            # Bitstream Vera - superb screen readability
    "DejaVu Sans",             # Common in Linux GUIs - very clear letterforms
    "DejaVu Sans Mono",        # Programmer font - distinct character shapes (OCR loves this)
    
    # === POSTSCRIPT STANDARD FONTS (Professional publishing) ===
    "Nimbus Roman No9 L",      # High-quality Times alternative - different hinting than Liberation
    "Nimbus Sans L",           # Helvetica clone - tight spacing tests OCR segmentation
    "Nimbus Mono L",           # Courier alternative - IBM typewriter style
    
    # === LATEX/ACADEMIC FONTS (Computer Modern family) ===
    "Latin Modern Roman",      # Computer Modern - distinctive academic/thesis font
    "Latin Modern Sans",       # CM Sans - geometric, wide proportions
    "Latin Modern Mono",       # CM Mono - very light, thin strokes (OCR stress test)
    
    # === ADOBE PROFESSIONAL FONTS ===
    "Source Serif Pro",        # Modern readable serif - high x-height
    "Source Sans Pro",         # UI-optimized - extremely clear at small sizes
    "Source Code Pro",         # Modern monospace - distinct l/1/I, 0/O
    
    # === MODERN WEB FONTS ===
    "Open Sans",               # Humanist sans - open apertures, very OCR-friendly
    "Roboto",                  # Android font - geometric with Grotesque features
    "Cantarell",               # GNOME font - rounded terminals, distinctive curves
    
    # === DISTINCTIVE SERIF STYLES ===
    "Bitstream Charter",       # Old-style/Baskerville style - different from Times
    "Century Schoolbook L",    # High x-height, schoolbook style - extremely OCR-readable
    "URW Bookman L",           # Light, wide serif - tests OCR on thin strokes
    "URW Palladio L",          # Palatino clone - elegant, different proportions than Times
    "Utopia",                  # Adobe transitional serif - distinctive sharp serifs
    
    # === MODERN SANS ===
    "Carlito",                 # Calibri clone - modern rounded sans (Microsoft Office standard)
    
    # === CHALLENGING BUT READABLE (Stress test) ===
    "URW Chancery L"           # Script/italic calligraphy - tests OCR italic handling
]

# Fallback font if none are found
FALLBACK_FONT = "DejaVuSans.ttf"  # Usually available on Linux

# ============================================================================
# CONFIGURATION
# ============================================================================
CANVAS_SIZE = 1024
OUTPUT_DIR = Path("./ocr_benchmark_images")
SCALES = [i / 10 for i in range(10, 0, -1)]  # 1.0, 0.9, ..., 0.1

# Aspect ratios from extreme width to extreme height
# 16:1, 8:1, 4:1, 2:1, 1:1, 1:2, 1:4, 1:8, 1:16
ASPECT_RATIOS = [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]


# LaTeX to plain text conversion (simplified)
def preprocess_text(text):
    """Convert LaTeX syntax to plain text for rendering without LaTeX engine."""
    import re
    # Remove section markers (## 1 Introduction → Introduction)
    text = re.sub(r'## \d+ ', '', text)
    # Convert $...$ to plain text (remove math delimiters but keep content)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    # Remove escaped parentheses \( and \)
    text = re.sub(r'\\\(|\\\)', '', text)
    # Replace LaTeX commands
    text = text.replace('\\ldots', '...')
    text = text.replace('\\dots', '...')
    text = text.replace('\\mathbf{', '')
    text = text.replace('\\', ' ')  # Replace remaining backslashes with space
    return text

CLEAN_TEXT = preprocess_text(SAMPLE_TEXT)


class TextRenderer:
    def __init__(self, font_name, font_size):
        self.font_name = font_name
        self.font_size = font_size
        self.font = self._load_font()

    def _load_font(self):
        """Attempt to load font, falling back to system defaults."""
        try:
            # Try to load by name (Windows/macOS)
            return ImageFont.truetype(self.font_name, self.font_size)
        except:
            try:
                # Try common Linux paths
                linux_paths = [
                    f"/usr/share/fonts/truetype/{self.font_name.replace(' ', '')}/{self.font_name.replace(' ', '')}.ttf",
                    f"/usr/share/fonts/TTF/{self.font_name}.ttf",
                    f"/System/Library/Fonts/{self.font_name}.ttf",
                    f"/Library/Fonts/{self.font_name}.ttf",
                    f"C:/Windows/Fonts/{self.font_name.replace(' ', '')}.ttf",
                ]
                for path in linux_paths:
                    if os.path.exists(path):
                        return ImageFont.truetype(path, self.font_size)
                raise
            except:
                # Final fallback
                try:
                    return ImageFont.truetype(FALLBACK_FONT, self.font_size)
                except:
                    return ImageFont.load_default()

    def get_text_dimensions(self, text, max_width):
        """Calculate dimensions of text when wrapped to max_width."""
        lines = []
        paragraphs = text.split("\n")

        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append("")
                continue
            words = paragraph.split()
            current_line = words[0] if words else ""

            for word in words[1:]:
                test_line = current_line + " " + word
                bbox = self.font.getbbox(test_line)
                if bbox[2] <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)

        # Calculate height
        if not lines:
            return 0, 0

        bbox = self.font.getbbox("Ay")  # Representative characters
        line_height = bbox[3] - bbox[1]
        total_height = len(lines) * line_height * 1.2  # 1.2 line spacing

        # Max width actually used
        max_line_width = max(self.font.getbbox(line)[2] for line in lines if line) if any(lines) else 0

        return max_line_width, total_height, lines


def find_max_font_size(font_name, canvas_size=CANVAS_SIZE, min_size=8, max_size=200):
    """
    Binary search for the largest font size where text fits in canvas_size x canvas_size.
    Returns the font size.
    """
    best_size = min_size

    for size in [min_size, max_size]:  # Quick bounds check
        renderer = TextRenderer(font_name, size)
        w, h, _ = renderer.get_text_dimensions(CLEAN_TEXT, canvas_size)
        if h <= canvas_size and w <= canvas_size:
            best_size = size

    if best_size == max_size:
        return max_size  # Even max fits

    # Binary search
    low, high = min_size, max_size
    while low <= high:
        mid = (low + high) // 2
        renderer = TextRenderer(font_name, mid)
        w, h, _ = renderer.get_text_dimensions(CLEAN_TEXT, canvas_size)

        if h <= canvas_size and w <= canvas_size:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_size

def generate_image(font_name, font_size, aspect_ratio, scale_pct, output_path):
    """
    Generate one benchmark image.
    
    Strategy:
    1. Calculate target geometry (width x height) based on scale and aspect ratio
    2. Create 1024x1024 canvas
    3. Wrap text to fit target_width
    4. Center text on canvas (both horizontally and vertically)
    5. Draw directly - NO upscaling/resizing to avoid blur
    """
    import re
    
    # Calculate target geometry
    # Scale applies to linear dimensions (font size and box dimensions)
    linear_scale = scale_pct / 100.0
    target_area = (CANVAS_SIZE * linear_scale) ** 2
    
    # For aspect_ratio = width/height
    # width * height = target_area
    height = math.sqrt(target_area / aspect_ratio)
    width = target_area / height
    
    # Clamp to canvas size
    width = min(int(width), CANVAS_SIZE)
    height = min(int(height), CANVAS_SIZE)
    
    # Create final canvas directly (no temp image, no resizing)
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(img)
    
    # Load font at the specified (already scaled) size
    try:
        font = TextRenderer(font_name, font_size).font
    except:
        font = ImageFont.load_default()

    # Word wrap text to target_width
    lines = []
    paragraphs = CLEAN_TEXT.split("\n")
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            lines.append("")
            continue
            
        words = paragraph.split()
        if not words:
            continue
            
        current_line = words[0]
        
        for word in words[1:]:
            test_line = current_line + " " + word
            bbox = font.getbbox(test_line)
            if bbox[2] <= width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        lines.append("")  # Paragraph break
    
    # Calculate actual text dimensions
    if not lines:
        img.save(output_path, "PNG")
        return output_path
        
    bbox_sample = font.getbbox("Ay")
    line_height = (bbox_sample[3] - bbox_sample[1]) * 1.2
    text_width = max(font.getbbox(line)[2] for line in lines if line) if any(lines) else 0
    text_height = len(lines) * line_height
    
    # Center text on canvas
    x_offset = (CANVAS_SIZE - min(text_width, width)) // 2
    y_offset = (CANVAS_SIZE - min(text_height, height)) // 2
    
    # Ensure non-negative offsets
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)
    
    # Draw text centered
    y = y_offset
    for line in lines:
        if y > CANVAS_SIZE:
            break
        if line:  # Only draw non-empty lines
            bbox_line = font.getbbox(line)
            line_w = bbox_line[2] - bbox_line[0]
            # Center each line individually for cleaner look
            x = (CANVAS_SIZE - line_w) // 2
            draw.text((x, int(y)), line, font=font, fill="black")
        y += line_height
    
    img.save(output_path, "PNG")
    return output_path

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Preprocess text once
    print(f"Text length: {len(CLEAN_TEXT)} characters")
    print(
        f"Generating {len(FONTS)} fonts × {len(SCALES)} scales × {len(ASPECT_RATIOS)} ratios = {len(FONTS)*len(SCALES)*len(ASPECT_RATIOS)} images"
    )

    total_images = len(FONTS) * len(SCALES) * len(ASPECT_RATIOS)
    pbar = tqdm(total=total_images, desc="Generating images")

    for font_name in FONTS:
        font_dir = OUTPUT_DIR / font_name.replace(" ", "_")
        font_dir.mkdir(exist_ok=True)

        print(f"\nProcessing font: {font_name}")

        # Find max font size for this font
        max_size = find_max_font_size(font_name)
        print(f"  Max font size for {font_name}: {max_size}pt")

        for scale in SCALES:
            scale_pct = int(scale * 100)
            font_size = int(max_size * scale)

            for ar in ASPECT_RATIOS:
                # Create filename: font_scale100_ar16.0.png
                ar_str = f"{ar:.2f}".replace(".", "p")
                filename = f"{font_name.replace(' ', '_')}_scale{scale_pct:03d}_ar{ar_str}.png"
                output_path = font_dir / filename

                try:
                    generate_image(font_name, font_size, ar, scale_pct, output_path)
                except Exception as e:
                    print(f"Error generating {filename}: {e}")
                    # Create blank image as placeholder
                    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
                    img.save(output_path)

                pbar.update(1)

    pbar.close()
    print(f"\nDone! Images saved to {OUTPUT_DIR.absolute()}")
    print(f"Total images generated: {total_images}")


if __name__ == "__main__":
    main()
