#!/usr/bin/env python3
"""
OCR Benchmark Image Generator - Debug Version
"""

import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

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
CANVAS_SIZE = 1024
OUTPUT_DIR = Path("./ocr_benchmark_images")
SCALES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ASPECT_RATIOS = [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]

def preprocess(text):
    import re
    text = re.sub(r'## \d+ ', '', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    text = re.sub(r'\\[()]', '', text)
    text = text.replace('\\ldots', '...').replace('\\dots', '...')
    text = text.replace('\\mathbf{', '').replace('\\', ' ')
    return text

CLEAN_TEXT = preprocess(SAMPLE_TEXT)

def load_font(name, size):
    """Load font with openSUSE Leap 15.6 specific paths."""
    import os
    import subprocess
    
    # Complete font path mapping for openSUSE Leap 15.6
    font_map = {
        # Liberation fonts
        "Liberation Serif": "/usr/share/fonts/truetype/LiberationSerif-Regular.ttf",
        "Liberation Sans": "/usr/share/fonts/truetype/LiberationSans-Regular.ttf",
        "Liberation Mono": "/usr/share/fonts/truetype/LiberationMono-Regular.ttf",
        
        # DejaVu fonts (common openSUSE locations)
        "DejaVu Serif": "/usr/share/fonts/truetype/DejaVuSerif.ttf",
        "DejaVu Sans": "/usr/share/fonts/truetype/DejaVuSans.ttf",
        "DejaVu Sans Mono": "/usr/share/fonts/truetype/DejaVuSansMono.ttf",
        
        # Nimbus fonts (TrueType versions)
        "Nimbus Roman No9 L": "/usr/share/fonts/truetype/NimbusRomNo9L-Regu.ttf",
        "Nimbus Sans L": "/usr/share/fonts/truetype/NimbusSanL-Regu.ttf",
        "Nimbus Mono L": "/usr/share/fonts/truetype/NimbusMonL-Regu.ttf",
        
        # Latin Modern (TeX Live)
        "Latin Modern Roman": "/usr/share/fonts/texlive-lm/lmroman10-regular.otf",
        "Latin Modern Sans": "/usr/share/fonts/texlive-lm/lmsans10-regular.otf",
        "Latin Modern Mono": "/usr/share/fonts/texlive-lm/lmmono10-regular.otf",
        
        # Adobe Source fonts
        "Source Serif Pro": "/usr/share/fonts/truetype/SourceSerifPro-Regular.otf",
        "Source Sans Pro": "/usr/share/fonts/truetype/SourceSansPro-Regular.otf",
        "Source Code Pro": "/usr/share/fonts/truetype/SourceCodePro-Regular.otf",
        
        # Web fonts
        "Open Sans": "/usr/share/fonts/truetype/OpenSans-Regular.ttf",
        "Roboto": "/usr/share/fonts/truetype/Roboto-Regular.ttf",
        "Cantarell": "/usr/share/fonts/truetype/Cantarell-VF.otf",
        
        # Distinctive serifs
        "Bitstream Charter": "/usr/share/fonts/ghostscript/bchr.pfa",
        "Century Schoolbook L": "/usr/share/fonts/truetype/CenturySchL-Roma.ttf",
        "URW Bookman L": "/usr/share/fonts/truetype/URWBookmanL-Ligh.ttf",
        "URW Palladio L": "/usr/share/fonts/truetype/URWPalladioL-Roma.ttf",
        "Utopia": "/usr/share/fonts/truetype/Utopia-Regular.ttf",
        "Carlito": "/usr/share/fonts/truetype/Carlito-Regular.ttf",
        
        # Chancery (italic script)
        "URW Chancery L": "/usr/share/fonts/truetype/URWChanceryL-MediItal.ttf",
    }
    
    # Try direct path first
    if name in font_map:
        path = font_map[name]
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass  # Fall through to fc-list
    
    # Fallback: use fc-list to find the font
    try:
        result = subprocess.run(
            ['fc-list', f':family={name}:style=Regular', '-f', '%{file}'],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            # Try without Regular style constraint
            result = subprocess.run(
                ['fc-list', f':family={name}', '-f', '%{file}'],
                capture_output=True, text=True
            )
        
        if result.stdout.strip():
            # Get first available file
            font_file = result.stdout.strip().split('\n')[0]
            if os.path.exists(font_file):
                return ImageFont.truetype(font_file, size)
    except:
        pass
    
    print(f"    WARNING: Could not load {name} at {size}pt, using default")
    return ImageFont.load_default()    
def wrap_text(font, text, max_width):
    """Return wrapped lines and dimensions."""
    lines = []
    for para in text.split('\n'):
        if not para.strip():
            lines.append('')
            continue
        words = para.split()
        current = words[0] if words else ''
        for word in words[1:]:
            test = current + ' ' + word
            if font.getbbox(test)[2] <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
    
    if not lines:
        return [], 0, 0
    
    line_h = (font.getbbox("Ay")[3] - font.getbbox("Ay")[1]) * 1.2
    total_h = len(lines) * line_h
    max_w = max(font.getbbox(l)[2] for l in lines if l) if any(lines) else 0
    return lines, max_w, total_h

def find_max_size(font_name):
    low, high = 8, 1000
    best = 8
    target_height = CANVAS_SIZE * 0.95  # Aim for 95% fill
    
    while low <= high:
        mid = (low + high) // 2
        font = load_font(font_name, mid)
        _, _, h = wrap_text(font, CLEAN_TEXT, CANVAS_SIZE)
        
        if h <= CANVAS_SIZE:
            best = mid
            if h >= target_height:  # Close enough to full
                return mid  # Early exit if we hit sweet spot
            low = mid + 1
        else:
            high = mid - 1
    return best
def generate(font_name, font_size, ar, scale):
    linear_scale = scale / 100.0 if scale > 1 else scale  # Handle both 0.9 and 90
    if scale > 1:
        scale = int(scale)
    
    # Calculate box
    area = (CANVAS_SIZE * linear_scale) ** 2
    h = math.sqrt(area / ar)
    w = area / h
    w = min(int(w), CANVAS_SIZE)
    h = min(int(h), CANVAS_SIZE)
    
    font = load_font(font_name, font_size)
    lines, text_w, text_h = wrap_text(font, CLEAN_TEXT, w)
    
    # DEBUG INFO
    info = f"  Scale {scale}%, AR {ar}: Box {w}x{h}, Text {text_w:.0f}x{text_h:.0f}, Lines {len(lines)}"
    
    if text_h > h or text_w > w:
        return None, info + f" -> NO FIT (need height {text_h:.0f} > {h})"
    
    # Create image - draw directly at final size, no canvas padding
    # We create a temporary image just big enough for the text
    margin = 10
    img_w = int(min(text_w + margin*2, CANVAS_SIZE))
    img_h = int(min(text_h + margin*2, CANVAS_SIZE))    
    img = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(img)
    
    line_h = (font.getbbox("Ay")[3] - font.getbbox("Ay")[1]) * 1.2
    y = margin
    
    for line in lines:
        if line:
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            x = (img_w - line_w) // 2  # Center horizontally
            draw.text((x, int(y)), line, font=font, fill='black')
        y += line_h
    
    # Tight crop to actual text (remove any remaining margin)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    return img, info + f" -> Generated {img.size}"

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    for font_name in FONTS:
        print(f"\n{'='*60}")
        print(f"Font: {font_name}")
        print(f"{'='*60}")
        
        max_size = find_max_size(font_name)
        print(f"Max font size for 1024x1024: {max_size}pt")
        print(f"Text length: {len(CLEAN_TEXT)} chars")
        
        generated = 0
        failed = 0
        
        # Generate for all combinations
        for scale in SCALES:
            font_sz = int(max_size * scale)
            
            for ar in ASPECT_RATIOS:
                img, info = generate(font_name, font_sz, ar, int(scale*100))
                print(info)
                
                if img:
                    ar_str = f"{ar:.2f}".replace('.', 'p')
                    fname = f"{font_name.replace(' ', '_')}_scale{int(scale*100):03d}_ar{ar_str}_{img.size[0]}x{img.size[1]}.png"
                    img.save(OUTPUT_DIR / fname)
                    generated += 1
                else:
                    failed += 1
        
        print(f"\nSUMMARY: Generated {generated}, Failed {failed}")
        print("Hint: If Failed is high, the text is too long for small scales.")
        print("      Try using a shorter SAMPLE_TEXT (e.g., just one paragraph).")

if __name__ == "__main__":
    main()