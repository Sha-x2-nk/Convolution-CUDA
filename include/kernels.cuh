#pragma once

#include "errorCheckUtils.cuh"

#include <cuda_runtime.h>

#include <iostream>

// R= Radius of convolution.
// M, N = Dimension of A
// A = Input Matrix
// C = Result Matrix
// F = Filter Matrix
template <int M, int N, int R>
__global__ void conv2D_1(float *A, float *F, float *C){
    int outCol = (blockIdx.y * blockDim.y + threadIdx.y);
    int outRow = (blockIdx.x * blockDim.x + threadIdx.x);

    for (int r = -R; r <= R; ++r)
        for (int c = -R; c <= R; ++c)
            if (outRow + r >= 0 && outRow + r < M && outCol + c >= 0 && outCol + c < N)
                C[outRow * N + outCol] += A[(outRow + r) * N + (outCol + c)] * F[ (R + r) * (2*R + 1) + (R + c)];
}

template<int HEIGHT, int WIDTH, int RADIUS>
void callConv2D_1(float *d_A, float *d_F, float *d_C, float *h_C){
    dim3 block(16, 16);
    dim3 grid(HEIGHT / 16, WIDTH / 16);

    conv2D_1<HEIGHT, WIDTH, RADIUS><<<grid, block>>>(d_A, d_F, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL( cudaMemcpy(h_C, d_C, HEIGHT*WIDTH*sizeof(float), cudaMemcpyDeviceToHost) );
}

// ################################## COALESCED ACCESS
// R= Radius of convolution.
// M, N = Dimension of A
// A = Input Matrix
// C = Result Matrix
// F = Filter Matrix
template <int M, int N, int R>
__global__ void conv2D_2(float *A, float *F, float *C){
    int outRow = (blockIdx.y * blockDim.y + threadIdx.y);
    int outCol = (blockIdx.x * blockDim.x + threadIdx.x);

    for (int r = -R; r <= R; ++r)
        for (int c = -R; c <= R; ++c)
            if (outRow + r >= 0 && outRow + r < M && outCol + c >= 0 && outCol + c < N)
                C[outRow * N + outCol] += A[(outRow + r) * N + (outCol + c)] * F[ (R + r) * (2*R + 1) + (R + c)];
}

template<int HEIGHT, int WIDTH, int RADIUS>
void callConv2D_2(float *d_A, float *d_F, float *d_C, float *h_C){
    dim3 block(16, 16);
    dim3 grid(WIDTH / 16, HEIGHT / 16);

    conv2D_2<HEIGHT, WIDTH, RADIUS><<<grid, block>>>(d_A, d_F, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL( cudaMemcpy(h_C, d_C, HEIGHT*WIDTH*sizeof(float), cudaMemcpyDeviceToHost) );
}


// ####################################### CONSTANT MEMORY
__constant__ float F[ (2*3 + 1) * (2*3 + 1)]; // Constant Memory Declaration

// R= Radius of convolution.
// M, N = Dimension of A
// A = Input Matrix
// C = Result Matrix
// F = Filter Matrix ( constant )
template <int M, int N, int R>
__global__ void conv2D_3(float *A, float *C){
    int outRow = (blockIdx.y * blockDim.y + threadIdx.y);
    int outCol = (blockIdx.x * blockDim.x + threadIdx.x);

    for (int r = -R; r <= R; ++r)
        for (int c = -R; c <= R; ++c)
            if (outRow + r >= 0 && outRow + r < M && outCol + c >= 0 && outCol + c < N)
                C[outRow * N + outCol] += A[(outRow + r) * N + (outCol + c)] * F[ (R + r) * (2*R + 1) + (R + c)];
}

template<int HEIGHT, int WIDTH, int RADIUS>
void callConv2D_3(float *d_A, float *d_F, float *d_C, float *h_C){
    dim3 block(16, 16);
    dim3 grid(WIDTH / 16, HEIGHT / 16);
    // constant memory for filter
    CUDA_CALL( cudaMemcpyToSymbol(F, d_F, (2*RADIUS + 1)*(2*RADIUS + 1), 0, cudaMemcpyDeviceToDevice) );

    conv2D_3<HEIGHT, WIDTH, RADIUS><<<grid, block>>>(d_A, d_C);
    cudaDeviceSynchronize();
    CUDA_CALL( cudaMemcpy(h_C, d_C, HEIGHT*WIDTH*sizeof(float), cudaMemcpyDeviceToHost) );
}
