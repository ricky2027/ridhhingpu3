# Batch Image Processing at Scale with CUDA NPP

A GPU-accelerated pipeline that batch-processes hundreds of images using
**NVIDIA Performance Primitives (NPP)**. It demonstrates resize, color
conversion, box filtering, Canny edge detection, and thresholding at scale,
with per-image GPU timing and a CSV execution log.

---

## Code Organization

| Path | Description |
|------|-------------|
| `bin/` | Compiled binary (`npp_processor`); git-ignored |
| `data/input/` | Input images (generated or your own); git-ignored |
| `data/output/` | Processed results and verbose intermediate stages |
| `data/results/` | `execution_log.csv` timing log |
| `lib/` | Bundled `stb_image.h` / `stb_image_write.h` for image I/O |
| `src/` | CUDA/C++ sources |
| `src/npp_pipeline.{h,cu}` | `NppPipeline` — wraps the five NPP operations |
| `src/image_io.{h,cu}` | Load/save + device memory helpers (stb + CUDA) |
| `src/main.cu` | CLI batch processor: scan, process, time, log, summarize |
| `scripts/generate_test_images.py` | Creates 200 synthetic test images |
| `Makefile` | Build rules (`all`, `clean`, `run`) |
| `run.sh` | Build + generate data + run end to end |
| `INSTALL` | Prerequisites and install steps |
| `LICENSE` | MIT license (plus stb public-domain notice) |

---

## Supported Operations

| Operation | NPP function | Description |
|-----------|--------------|-------------|
| Resize | `nppiResize_8u_C3R` | Scales a 3-channel image to a target size (linear interp.) |
| Grayscale | `nppiRGBToGray_8u_C3C1R` | Converts 3-channel RGB to single-channel luma |
| Box filter (blur) | `nppiFilterBox_8u_C1R` | Mean blur on a grayscale image with an N×N mask |
| Canny edges | `nppiFilterCannyBorder_8u_C1R` | Edge map from low/high hysteresis thresholds |
| Threshold | `nppiThreshold_LTValGTVal_8u_C1R` | Binarizes grayscale to {0, 255} |

The `full` pipeline chains: **resize → grayscale → box filter → Canny edges**.

---

## Prerequisites

- CUDA Toolkit 11.0+ (includes `nvcc` and the NPP libraries)
- An NVIDIA GPU + driver (Linux x86_64)
- Python 3 + Pillow — only to generate synthetic test images

See [`INSTALL`](INSTALL) for details.

---

## Building

```bash
make
```

This produces `bin/npp_processor`. Run `make clean` to remove build artifacts.

---

## Usage

```
./bin/npp_processor -i <input_dir> [options]
```

| Flag | Meaning | Default |
|------|---------|---------|
| `-i <path>` | Input directory containing images (**required**) | — |
| `-o <path>` | Output directory | `data/output/` |
| `-p <pipeline>` | `resize`, `grayscale`, `blur`, `canny`, `threshold`, `full` | `full` |
| `-w <width>` | Target resize width (height keeps aspect ratio) | `512` |
| `-t <value>` | Threshold value for `threshold` / Canny high threshold | `128` |
| `-v` | Verbose: also save intermediate pipeline stages | off |
| `-h` | Print usage and exit | — |

### Examples

```bash
# Full pipeline with verbose intermediate stages
./bin/npp_processor -i data/input/ -o data/output/ -p full -v

# Resize every image to 256 px wide
./bin/npp_processor -i data/input/ -p resize -w 256

# Grayscale conversion only
./bin/npp_processor -i data/input/ -p grayscale

# 5x5 box blur
./bin/npp_processor -i data/input/ -p blur

# Canny edges with a high threshold of 100
./bin/npp_processor -i data/input/ -p canny -t 100

# Binarize at a threshold of 90
./bin/npp_processor -i data/input/ -p threshold -t 90
```

Each result is saved as `<original_name>_<operation>.png`. With `-v`, the full
pipeline additionally writes `<name>_1_resize.png`, `<name>_2_grayscale.png`,
`<name>_3_blur.png`, and `<name>_4_canny.png`.

---

## Quick Start

```bash
bash run.sh
```

This builds the binary, generates 200 synthetic test images into
`data/input/` if it is empty, runs the full pipeline, and writes the timing log
to `data/results/execution_log.csv`.

---

## Data

The demo is self-contained. `scripts/generate_test_images.py` creates 200
images across mixed resolutions (256², 512², 1024²) with four pattern types:

- **Gradients** — smooth two-color ramps
- **Noise** — uniform random RGB
- **Shapes** — rectangles, ellipses, and lines (good for edge detection)
- **Blocks** — grids of solid color

Generate them manually with:

```bash
python3 scripts/generate_test_images.py --count 200 --out data/input
```

To use **your own** images, just drop `.png`, `.jpg`, `.bmp`, or `.tif` files
into `data/input/` (or point `-i` at any directory).

---

## Results

`data/results/execution_log.csv` has one row per processed image:

| Column | Meaning |
|--------|---------|
| `image_name` | Base name (no extension) |
| `original_width` | Source width in pixels |
| `original_height` | Source height in pixels |
| `operation` | Pipeline that ran |
| `gpu_time_ms` | GPU time measured with CUDA events |
| `timestamp` | When the image was processed |

The program also prints a summary: total images processed, total GPU time,
average time per image, and throughput (images/sec).

Sample timing numbers (fill in after running on your hardware):

| GPU | Operation | Images | Avg ms/img | Throughput (img/s) |
|-----|-----------|-------:|-----------:|-------------------:|
| _e.g. RTX 3060_ | `full` | 200 | _TBD_ | _TBD_ |

Before/after samples (add your own screenshots):

```
data/input/test_0002_shapes_512.png   ->   data/output/test_0002_shapes_512_full.png
```

---

## Architecture

The pipeline flow for each image:

```
 load (stb)  ->  allocate device buffer (cudaMalloc)
             ->  upload (cudaMemcpy2D, host -> device)
             ->  process (NPP ops, timed with cudaEvent_t)
             ->  download result (device -> host)
             ->  save PNG (stb)  +  append CSV row
             ->  free device/host memory
```

`main.cu` drives the batch loop; `NppPipeline` owns the NPP calls (each wrapped
in `NPP_CHECK` for named error reporting); `image_io` centralizes stb I/O and
CUDA memory management behind `CUDA_CHECK`. A failure to load one image logs a
warning and skips it — the batch never crashes on a single bad file.

---

## Lessons Learned

_(Fill in after running the project.)_

- NPP step/pitch handling: ...
- Choosing Canny thresholds for synthetic vs. natural images: ...
- Where GPU time is dominated by transfer vs. compute: ...
