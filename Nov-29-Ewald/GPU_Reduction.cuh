//#include <stdio.h>
#include <iostream>
#include <cuda_fp16.h>
void checkCUDAError(const char *msg);
template<size_t blockSize, typename T>
T GPUReduction(T* dA, size_t N);
#define BLOCKSIZE 1024
#define DEFAULTTHREAD 128
inline void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        printf("CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE);
    }
}

template <size_t blockSize, typename T>
__device__ void warpReduce(volatile T *sdata, size_t tid)
{
    if (blockSize >= 64) sdata[tid] = sdata[tid] + sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] = sdata[tid] + sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] = sdata[tid] + sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] = sdata[tid] + sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] = sdata[tid] + sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] = sdata[tid] + sdata[tid +  1];
}

template <size_t blockSize, typename T>
__global__ void reduceCUDA(T* g_idata, T* g_odata, size_t n)
{
    __shared__ T sdata[blockSize];

    size_t tid = threadIdx.x;
    //size_t i = blockIdx.x*(blockSize*2) + tid;
    //size_t gridSize = blockSize*2*gridDim.x;
    size_t i = blockIdx.x*(blockSize) + tid;
    size_t gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
    //while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template<size_t blockSize, typename T>
T GPUReduction_Stream(T* dA, size_t N, cudaStream_t stream)
{
    T tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);

    T* tmp;
    cudaMallocAsync(&tmp, sizeof(T) * blocksPerGrid, stream); checkCUDAError("Error allocating tmp [GPUReduction]");

    T* from = dA;

    do
    {
        blocksPerGrid   = std::ceil((1.*n) / blockSize);
        reduceCUDA<blockSize><<<blocksPerGrid, blockSize, 0, stream>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1)
        reduceCUDA<blockSize><<<1, blockSize, 0, stream>>>(tmp, tmp, n);

    //cudaDeviceSynchronize();
    checkCUDAError("Error launching kernel [GPUReduction]");

    cudaMemcpyAsync(&tot, tmp, sizeof(T), cudaMemcpyDeviceToHost, stream); checkCUDAError("Error copying result [GPUReduction]");
    cudaFree(tmp);
    return tot;
}

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template<size_t blockSize, typename T>
T GPUReduction(T* dA, size_t N)
{
    T tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);

    T* tmp;
    cudaMalloc(&tmp, sizeof(T) * blocksPerGrid); checkCUDAError("Error allocating tmp [GPUReduction]");

    T* from = dA;

    do
    {
        blocksPerGrid   = std::ceil((1.*n) / blockSize);
        reduceCUDA<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1)
        reduceCUDA<blockSize><<<1, blockSize>>>(tmp, tmp, n);

    //cudaDeviceSynchronize();
    checkCUDAError("Error launching kernel [GPUReduction]");

    cudaMemcpy(&tot, tmp, sizeof(T), cudaMemcpyDeviceToHost); checkCUDAError("Error copying result [GPUReduction]");
    cudaFree(tmp);
    return tot;
}

template <size_t blockSize>
__device__ void half_warpReduce(volatile half *sdata, size_t tid)
{
    if (blockSize >= 64) sdata[tid] = __hadd_rn(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = __hadd_rn(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = __hadd_rn(sdata[tid], sdata[tid +  8]);
    if (blockSize >=  8) sdata[tid] = __hadd_rn(sdata[tid], sdata[tid +  4]);
    if (blockSize >=  4) sdata[tid] = __hadd_rn(sdata[tid], sdata[tid +  2]);
    if (blockSize >=  2) sdata[tid] = __hadd_rn(sdata[tid], sdata[tid +  1]);
}

template <size_t blockSize>
__global__ void half_reduceCUDA(half* g_idata, half* g_odata, size_t n)
{
    __shared__ half sdata[blockSize];

    size_t tid = threadIdx.x;
    //size_t i = blockIdx.x*(blockSize*2) + tid;
    //size_t gridSize = blockSize*2*gridDim.x;
    size_t i = blockIdx.x*(blockSize) + tid;
    size_t gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
    //while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = __hadd_rn(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] = __hadd_rn(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] = __hadd_rn(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] = __hadd_rn(sdata[tid], sdata[tid +  64]); } __syncthreads(); }

    if (tid < 32) half_warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template<size_t blockSize>
half half_GPUReduction(half* dA, size_t N)
{
    half tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);

    half* tmp;
    cudaMalloc(&tmp, sizeof(half) * blocksPerGrid); checkCUDAError("Error allocating tmp [GPUReduction]");

    half* from = dA;

    do
    {
        blocksPerGrid   = std::ceil((1.*n) / blockSize);
        half_reduceCUDA<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1)
        half_reduceCUDA<blockSize><<<1, blockSize>>>(tmp, tmp, n);

    //cudaDeviceSynchronize();
    checkCUDAError("Error launching kernel [GPUReduction]");

    cudaMemcpy(&tot, tmp, sizeof(half), cudaMemcpyDeviceToHost); checkCUDAError("Error copying result [GPUReduction]");
    cudaFree(tmp);
    return tot;
}
