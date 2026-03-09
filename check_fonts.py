#!/usr/bin/env python3
"""Check which fonts are available on openSUSE."""

import subprocess

FONTS = [
    "Liberation Serif", "Liberation Sans", "Liberation Mono",
    "DejaVu Serif", "DejaVu Sans", "DejaVu Sans Mono",
    "Nimbus Roman No9 L", "Nimbus Sans L", "Nimbus Mono L",
    "Latin Modern Roman", "Latin Modern Sans", "Latin Modern Mono",
    "Source Serif Pro", "Source Sans Pro", "Source Code Pro",
    "Open Sans", "Roboto", "Cantarell",
    "Bitstream Charter", "Century Schoolbook L", "URW Bookman L",
    "URW Palladio L", "Utopia", "Carlito", "URW Chancery L"
]

print("Checking font availability...\n")
found = []
missing = []

for font in FONTS:
    # Correct fc-list syntax
    result = subprocess.run(
        ['fc-list', f':family={font}'],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        # Get the file path for Regular style
        file_result = subprocess.run(
            ['fc-list', f':family={font}:style=Regular', '-f', '%{file}'],
            capture_output=True, text=True
        )
        path = file_result.stdout.strip().split('\n')[0] if file_result.stdout else "Unknown"
        print(f"✓ {font:<25} -> {path}")
        found.append((font, path))
    else:
        print(f"✗ {font:<25} -> NOT FOUND")
        missing.append(font)

print(f"\n{'='*60}")
print(f"Found: {len(found)}/{len(FONTS)}")
if missing:
    print(f"\nMissing: {missing}")