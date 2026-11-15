#!/usr/bin/env python3
"""Generate synthetic test images for the CUDA NPP batch processor.

Creates a configurable number of images (default 200) with a variety of
patterns -- gradients, random noise, geometric shapes, and solid color blocks
-- at mixed resolutions (256x256, 512x512, 1024x1024). This keeps the batch
processing demo self-contained: no manual image downloads required.

Usage:
    python3 scripts/generate_test_images.py [--count N] [--out DIR]

Requires Pillow:  pip install Pillow
"""

import argparse
import os
import random

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - guidance for the user
    raise SystemExit(
        "Pillow is required. Install it with:  pip install Pillow"
    )


SIZES = [256, 512, 1024]


def random_color():
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def make_gradient(size):
    """Horizontal gradient between two random colors."""
    img = Image.new("RGB", (size, size))
    c0 = random_color()
    c1 = random_color()
    px = img.load()
    for x in range(size):
        t = x / max(1, size - 1)
        r = int(c0[0] + (c1[0] - c0[0]) * t)
        g = int(c0[1] + (c1[1] - c0[1]) * t)
        b = int(c0[2] + (c1[2] - c0[2]) * t)
        for y in range(size):
            px[x, y] = (r, g, b)
    return img


def make_noise(size):
    """Uniform random RGB noise."""
    data = bytes(random.getrandbits(8) for _ in range(size * size * 3))
    return Image.frombytes("RGB", (size, size), data)


def make_shapes(size):
    """Random geometric shapes on a solid background -- good for edge tests."""
    img = Image.new("RGB", (size, size), random_color())
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(5, 15)):
        x0, y0 = random.randint(0, size), random.randint(0, size)
        x1, y1 = random.randint(0, size), random.randint(0, size)
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        shape = random.choice(("rectangle", "ellipse", "line"))
        if shape == "rectangle":
            draw.rectangle(box, fill=random_color(), outline=(0, 0, 0))
        elif shape == "ellipse":
            draw.ellipse(box, fill=random_color(), outline=(0, 0, 0))
        else:
            draw.line(box, fill=random_color(), width=random.randint(1, 8))
    return img


def make_blocks(size):
    """Grid of solid color blocks."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    cells = random.choice((2, 4, 8))
    step = size // cells
    for i in range(cells):
        for j in range(cells):
            box = [i * step, j * step, (i + 1) * step, (j + 1) * step]
            draw.rectangle(box, fill=random_color())
    return img


GENERATORS = [make_gradient, make_noise, make_shapes, make_blocks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200,
                        help="number of images to generate (default: 200)")
    parser.add_argument("--out", default="data/input",
                        help="output directory (default: data/input)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    for i in range(args.count):
        size = random.choice(SIZES)
        generator = GENERATORS[i % len(GENERATORS)]
        img = generator(size)
        name = f"test_{i:04d}_{generator.__name__[5:]}_{size}.png"
        img.save(os.path.join(args.out, name))
        if (i + 1) % 25 == 0 or i + 1 == args.count:
            print(f"  generated {i + 1}/{args.count} images")

    print(f"Done. {args.count} images written to {args.out}/")


if __name__ == "__main__":
    main()
