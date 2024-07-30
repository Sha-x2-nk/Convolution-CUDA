#include "include/errorCheckUtils.cuh"
#include "include/kernels.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <random>

const int HEIGHT = 512;
const int WIDTH = 512;
const int RADIUS = 3;
const int FILTER_SIZE = 2 * RADIUS + 1;
const int IN_CHANNELS = 1;
const int OUT_CHANNELS = 1;
const int STRIDE = 1;
const int PADDING = RADIUS;

void initMat(float *A, int M, int N){
    for (int i = 0; i < M * N; ++i)
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

int main()
{

    // allocating host arrays
    float *h_A = (float *)malloc(HEIGHT * WIDTH * sizeof(float));
    initMat(h_A, HEIGHT, WIDTH);

    float *h_C = (float *)malloc(HEIGHT * WIDTH * sizeof(float));

    float *h_F = (float *)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));
    initMat(h_F, FILTER_SIZE, FILTER_SIZE);

    // allocating device arrays
    float *d_A, *d_F, *d_C;
    CUDA_CALL( cudaMalloc((void **)&d_A, HEIGHT*WIDTH*sizeof(float)) );
    CUDA_CALL( cudaMalloc((void **)&d_C, HEIGHT*WIDTH*sizeof(float)) );
    CUDA_CALL( cudaMalloc((void **)&d_F, FILTER_SIZE*FILTER_SIZE*sizeof(float)) );

    // copying from host to device
    CUDA_CALL( cudaMemcpy(d_A, h_A, HEIGHT*WIDTH*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_F, h_F, FILTER_SIZE*FILTER_SIZE*sizeof(float), cudaMemcpyHostToDevice) );

    callConv2D_1<HEIGHT, WIDTH, RADIUS>(d_A, d_F, d_C, h_C);
    callConv2D_2<HEIGHT, WIDTH, RADIUS>(d_A, d_F, d_C, h_C);
    callConv2D_3<HEIGHT, WIDTH, RADIUS>(d_A, d_F, d_C, h_C);
}