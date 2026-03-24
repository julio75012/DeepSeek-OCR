#!/usr/bin/env python3
"""
OCR Benchmark Image Generator - Ubuntu Fixed Version (25 Fonts)
Bug fix: Preserves paragraph breaks (empty lines)
"""

import os
import math
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Read ground truth text from file
GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.txt"
with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
    SAMPLE_TEXT = f.read()

# High-Entropy OCR Benchmark Font Set (25 fonts)
FONTS = [
    # === SERIF FONTS (11) ===
    "Liberation Serif",
    "DejaVu Serif",
    "FreeSerif",
    "Gentium",
    "Gentium Book Basic",
    "GentiumAlt",
    "Gentium Basic",
    "Linux Libertine O",
    "Linux Libertine Display O",
    "Bitstream Charter",
    "DejaVu Math TeX Gyre",
    "Caladea",

    # === SANS-SERIF FONTS (8) ===
    "Liberation Sans",
    "DejaVu Sans",
    "FreeSans",
    "Carlito",
    "Linux Biolinum O",
    "Loma",
    "Liberation Sans Narrow",
    "IPAPGothic",

    # === MONOSPACE FONTS (4) ===
    "Liberation Mono",
    "DejaVu Sans Mono",
    "FreeMono",
    "Noto Mono",

    # === SPECIALIZED (1) ===
    "Courier 10 Pitch",
]

# Verified font paths for Ubuntu
FONT_PATH_CACHE = {
    "Liberation Serif": "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    "DejaVu Serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
    "FreeSerif": "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "Gentium": "/usr/share/fonts/truetype/gentium/Gentium-R.ttf",
    "Gentium Book Basic": "/usr/share/fonts/truetype/gentium-basic/GenBkBasR.ttf",
    "GentiumAlt": "/usr/share/fonts/truetype/gentium/GentiumAlt-R.ttf",
    "Gentium Basic": "/usr/share/fonts/truetype/gentium-basic/GenBasR.ttf",
    "Linux Libertine O": "/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf",
    "Linux Libertine Display O": "/usr/share/fonts/opentype/linux-libertine/LinLibertine_DR.otf",
    "Bitstream Charter": "/usr/share/fonts/X11/Type1/c0648bt_.pfb",
    "DejaVu Math TeX Gyre": "/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf",
    "Caladea": "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf",
    "Liberation Sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVu Sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "FreeSans": "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "Carlito": "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
    "Linux Biolinum O": "/usr/share/fonts/opentype/linux-libertine/LinBiolinum_R.otf",
    "Loma": "/usr/share/fonts/opentype/tlwg/Loma.otf",
    "Liberation Sans Narrow": "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
    "IPAPGothic": "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "Liberation Mono": "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "DejaVu Sans Mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "FreeMono": "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "Noto Mono": "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    "Courier 10 Pitch": "/usr/share/fonts/X11/Type1/c0419bt_.pfb",
}

CANVAS_SIZE = 1024
OUTPUT_DIR = Path("./ocr_benchmark_images")
SCALES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ASPECT_RATIOS = [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
ALIGNMENTS = ["center", "left"]

def load_font(name, size):
    """Load font with guaranteed path resolution. Raises error on failure."""
    if name in FONT_PATH_CACHE:
        path = FONT_PATH_CACHE[name]
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    try:
        result = subprocess.run(
            ['fc-list', f':family={name}', '-f', '%{file}'],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            path = result.stdout.strip().split('\n')[0]
            return ImageFont.truetype(path, size)
    except:
        pass

    raise ValueError(f"Font {name} not found.")

def preprocess(text):
    import re
    text = re.sub(r'## \d+ ', '', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    text = re.sub(r'\\[()]', '', text)
    text = text.replace('\\ldots', '...').replace('\\dots', '...')
    text = text.replace('\\mathbf{', '').replace('\\', ' ')
    return text

CLEAN_TEXT = preprocess(SAMPLE_TEXT)

def wrap_text(font, text, max_width):
    """Return wrapped lines preserving paragraph breaks (empty lines)."""
    lines = []
    paragraphs = text.split('\n')

    for i, para in enumerate(paragraphs):
        # Handle empty paragraphs (preserve them as blank lines)
        if not para.strip():
            lines.append('')  # Empty string represents blank line
            continue

        # Wrap non-empty paragraph
        words = para.split()
        if not words:
            lines.append('')
            continue

        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + ' ' + word
            bbox = font.getbbox(test_line)
            if bbox and bbox[2] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

    if not lines:
        return [], 0, 0

    # Calculate metrics
    bbox_ay = font.getbbox("Ay")
    if not bbox_ay:
        bbox_ay = (0, 0, 10, 24)  # Fallback
    line_height = (bbox_ay[3] - bbox_ay[1]) * 1.2

    # Calculate total height including empty lines
    total_height = len(lines) * line_height

    # Calculate max width (ignoring empty lines for width calc)
    max_line_width = 0
    for line in lines:
        if line:
            bbox = font.getbbox(line)
            if bbox and bbox[2] > max_line_width:
                max_line_width = bbox[2]

    return lines, max_line_width, total_height

def find_max_size(font_name):
    """Binary search for max font size that fits in CANVAS_SIZE."""
    low, high = 8, 1000
    best = 8

    while low <= high:
        mid = (low + high) // 2
        try:
            font = load_font(font_name, mid)
            _, _, h = wrap_text(font, CLEAN_TEXT, CANVAS_SIZE)

            if h <= CANVAS_SIZE:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        except Exception as e:
            high = mid - 1
    return best

def check_existing(font_name, scale, ar, alignment, dims):
    """Check if image already exists."""
    ar_str = f"{ar:.2f}".replace('.', 'p')
    fname = f"{font_name.replace(' ', '_')}_scale{int(scale*100):03d}_ar{ar_str}_align-{alignment}_{dims[0]}x{dims[1]}.png"
    fpath = OUTPUT_DIR / fname
    return fpath.exists(), fname, fpath

def generate(font_name, font_size, ar, scale, alignment="center"):
    """Generate image with preserved paragraph breaks."""
    linear_scale = scale / 100.0 if scale > 1 else scale

    # Calculate box dimensions
    area = (CANVAS_SIZE * linear_scale) ** 2
    h = math.sqrt(area / ar)
    w = area / h
    w = min(int(w), CANVAS_SIZE)
    h = min(int(h), CANVAS_SIZE)

    font = load_font(font_name, font_size)
    lines, text_w, text_h = wrap_text(font, CLEAN_TEXT, w)

    info = f"  Scale {scale}%, AR {ar}, Align {alignment}: Box {w}x{h}, Text {text_w:.0f}x{text_h:.0f}"

    if text_h > h or text_w > w:
        return None, info + " -> NO FIT"

    # Check if already exists
    dims_preview = (int(min(text_w + 20, CANVAS_SIZE)), int(min(text_h + 20, CANVAS_SIZE)))
    exists, fname, fpath = check_existing(font_name, scale, ar, alignment, dims_preview)
    if exists:
        return None, info + f" -> EXISTS (skipped {fname})"

    # Create image
    margin = 10
    img_w = int(min(text_w + margin*2, CANVAS_SIZE))
    img_h = int(min(text_h + margin*2, CANVAS_SIZE))
    img = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(img)

    # Calculate line height
    bbox_ay = font.getbbox("Ay")
    if not bbox_ay:
        bbox_ay = (0, 0, 10, 24)
    line_h = (bbox_ay[3] - bbox_ay[1]) * 1.2

    y = margin

    for line in lines:
        if line:  # Only draw non-empty lines
            bbox = font.getbbox(line)
            if bbox:
                line_w = bbox[2] - bbox[0]

                if alignment == "center":
                    x = (img_w - line_w) // 2
                else:
                    x = margin

                draw.text((x, int(y)), line, font=font, fill='black')

        # Always advance y, even for empty lines (preserves paragraph breaks)
        y += line_h

    # Tight crop
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Save
    final_fname = f"{font_name.replace(' ', '_')}_scale{int(scale*100):03d}_ar{ar:.2f}".replace('.', 'p')
    final_fname += f"_align-{alignment}_{img.size[0]}x{img.size[1]}.png"
    final_path = OUTPUT_DIR / final_fname
    img.save(final_path)

    return img, info + f" -> Generated {final_fname}"

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("OCR Benchmark Generator - 25 Font High-Entropy Set")
    print("=" * 70)
    print(f"Fonts: {len(FONTS)} verified working fonts")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print("=" * 70)

    for font_name in FONTS:
        print(f"\nFont: {font_name}")

        try:
            max_size = find_max_size(font_name)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Max font size: {max_size}pt")

        generated = 0
        skipped = 0
        failed = 0

        for scale in SCALES:
            font_sz = int(max_size * scale)

            for ar in ASPECT_RATIOS:
                for alignment in ALIGNMENTS:
                    try:
                        img, info = generate(font_name, font_sz, ar, scale, alignment)
                        print(info)

                        if img:
                            generated += 1
                        elif "EXISTS" in info:
                            skipped += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        failed += 1

        print(f"  Summary: Generated {generated}, Skipped {skipped}, Failed {failed}")

if __name__ == "__main__":
    main()