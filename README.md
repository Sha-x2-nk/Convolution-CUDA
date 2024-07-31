# Optimising Direct Convolution kernels on CUDA
## Introduction
This project implements three convolution kernels using CUDA (Compute Unified Device Architecture) with optimizations for improved performance. Convolution kernels are fundamental operations in deep learning and image processing, and optimizing them for GPU execution can significantly accelerate computation.

## Kernels and Performance
#### HARDWARE: RTX 3070Ti ( Compute Capablity 8.6 )
| kernel | runtime | Relative Speedup | Absolute Speedup |
|--------|---------|-------------|--|
|1. Basic Kernel | 1.61 ms | 1 x | 1x |
|2. Memory Coalescing | 98 us | 16.4 x | 16.4 x|
|3. Constant Memory | 91 us | 1.07 x | 17.69 x|


## Usage
* Compile using nvcc

    <code>nvcc main.cu -o main.exe</code>

* Run
    <code>main.exe</code>

* Tune parameters like BLOCK_SIZE for your hardware.

## Acknowledgements
* Programming Massively Parallel Processors (Wen-mei W. Huw, David B. Kirk, Izzat El Hajj)
