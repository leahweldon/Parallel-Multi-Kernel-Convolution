# Parallel Multi-Kernel Convolution

**Parallel CPU convolution using OpenMP, SSE SIMD intrinsics, and cache-conscious kernel access.**

A collaborative systems-programming project for **CSU33014 - Concurrent Systems I** at **Trinity College Dublin (2023–2024)**. We optimised a multi-channel, multi-kernel convolution routine used to model the computational core of a convolutional neural network (CNN) layer. The goal was to produce the same result as a supplied scalar reference while reducing execution time on an x86 multicore machine.

**Team:** Faith Olopade · Darius Pop · Leah Weldon  
**Language:** C  
**Techniques:** OpenMP · SIMD (SSE) · kernel-data reordering · performance measurement

## Highlights

- **Kernel-level parallelism:** OpenMP distributes independent output-kernel computations among CPU threads.
- **SIMD vectorisation:** 128-bit SSE intrinsics process four `float` products per iteration and accumulate partial sums in double precision.
- **Kernel repacking:** Each worker builds a local, channel-contiguous floating-point representation of its integer-valued kernel to improve access through the inner accumulation loop.
- **Reference comparison:** The supplied harness times both the scalar implementation and the optimised routine and checks the sum of absolute output differences.
- **Historical results:** The team's report records speedups ranging from **6.21× to 98.27×** for its listed configurations. These are **2024 coursework measurements**, not performance guarantees or independently reproduced benchmarks.

## Problem and implementation

A conventional convolutional layer applies multiple filters over an input tensor with multiple channels. Each output value involves contributions from every input channel and every spatial position of its corresponding kernel. With large channel and filter counts, a direct implementation performs substantial repeated arithmetic and memory access.

The supplied benchmark represents the input image as `image[width][height][channels]` and the filter bank as `kernels[filters][channels][kernel_x][kernel_y]`. The original reference routine iterates over filters, image positions, channels, and kernel coordinates.

The collaborative implementation, located in [`src/conv-harness.c`](src/conv-harness.c), preserves the original harness structure and changes the compute path in `student_conv()`:

1. OpenMP assigns each output filter to a separate iteration of `#pragma omp parallel for`.
2. The filter weights are converted from `int16_t` to `float` and reordered as `[kernel_x][kernel_y][channel]` for channel-contiguous access.
3. Within each output pixel, the code processes 32 channels per outer step, using eight four-lane SSE multiplications.
4. Product lanes are converted into double-precision SIMD accumulators, then reduced to the final output value.

The program uses 128-bit SSE intrinsics such as `_mm_loadu_ps`, `_mm_mul_ps`, `_mm_cvtps_pd` and `_mm_hadd_pd`.

## Results from the original coursework

| Image | Kernel | Channels | Filters | Reference | Optimised | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| 16 × 16 | 7 | 32 | 32 | 48.3 ms | 7.8 ms | **6.21×** |
| 128 × 128 | 3 | 128 | 32 | 1.79 s | 28.0 ms | **63.78×** |
| 128 × 128 | 3 | 128 | 2,048 | 90.4 s | 939.2 ms | **96.25×** |
| 128 × 128 | 7 | 256 | 256 | 134.3 s | 1.37 s | **98.27×** |
| 512 × 512 | 1 | 512 | 512 | 418.1 s | 14.58 s | **28.67×** |

The complete 13-configuration transcription is in [`results/historical-benchmarks.csv`](results/historical-benchmarks.csv), sourced from the 2024 report and runtime screenshot. Figures are rounded for presentation. The report identifies Trinity's multicore `stoker.scss.tcd.ie` as its test environment. Hardware, thread count, load, compiler settings and data shape all affect observed speedup; new runs are not directly comparable without controlling these factors.

See the [public copy of the original team report](docs/original-group-report-public.pdf) for further historical context. Contributor names are retained; personal student identifiers and email addresses have been omitted from that copy.

## Build and run

**Requirements:** x86/x86-64 Linux or a compatible development environment; GCC with SSE support; OpenMP runtime (`libgomp`); `make`. This source uses Intel SSE intrinsics and is **not a native Apple Silicon / ARM build**.

```bash
make
OMP_NUM_THREADS=4 ./conv-harness 16 16 3 32 32
```

Command-line parameters are, in order:

```text
./conv-harness <image-width> <image-height> <kernel-order> <channels> <filters>
```

Supported kernel orders are 1, 3, 5 and 7. The implementation's inner loop assumes a channel count divisible by 32; use channel counts that are powers of two in the coursework range (32–2,048). The supplied workload considers image dimensions from 16 to 512 and filter counts from 32 to 2,048. Start with small inputs: the full scalar reference can be expensive at the upper limits.

To run four small reference-comparison checks:

```bash
make test
```

The harness prints elapsed times and an output-difference check. The test script rejects a reported reference mismatch; it is a **smoke test**, not proof of exhaustive correctness or reproducibility of the historical speedups. Input generation uses a time-derived random seed.

## Project structure

```text
.
├── src/
│   └── conv-harness.c                  # Reference harness and team optimised routine
├── docs/
│   ├── original-group-report-public.pdf
│   └── PROVENANCE.md
├── results/
│   └── historical-benchmarks.csv
├── scripts/
│   └── smoke-test.sh
├── Makefile
├── .gitignore
└── README.md
```

## Authorship and provenance

**Team project:** Faith Olopade, Darius Pop and Leah Weldon. The original test harness and scalar reference carry attribution to **Dr David Gregg (March 2022)**; the student implementation and 2024 analysis were developed as collaborative coursework.

This archive is an independent presentation of the same **shared project**, not a claim of sole authorship. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for source-retention, privacy and benchmark details.
