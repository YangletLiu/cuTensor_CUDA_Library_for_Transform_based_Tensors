#include "gemmStrideBatched.h"

void gemmStrideBatched(cuComplex *A, cuComplex *B, cuComplex *C, int Am, int An, int Ak,  int Bn) {
    cublasHandle_t handle;
    cuComplex alpha;
    alpha.x =1;
    alpha.y =0;
    cuComplex beta;
    beta.x = 0;
    beta.y = 0;
    int Bm = An;
    int Bk = Ak;
    int strA = Am*An;
    int strB = Bm*Bn;
    int strC = Am*Bn;
    int batchCount = Ak;

    cuComplex *d_A, *d_B, *d_C;
    cublasCreate(&handle);
    cudaMalloc ((void**)&d_A, sizeof(cuComplex) * Am*An*Ak);
    cudaMalloc ((void**)&d_B, sizeof(cuComplex) * Bm*Bn*Bk);
    cudaMalloc ((void**)&d_C, sizeof(cuComplex) * Am*Bn*Ak);

    cudaMemcpy(d_A, A, sizeof(cuComplex) * Am*An*Ak, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(cuComplex) * Bm*Bn*Bk, cudaMemcpyHostToDevice);

    cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, Am, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);

    cublasDestroy(handle);
    cudaMemcpy(C, d_C, sizeof(cuComplex) * Am*Bn*Ak, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

