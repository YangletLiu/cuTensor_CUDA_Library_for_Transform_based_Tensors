#include "tprod.h"
void gemmStrideBased(cuComplex* A, cuComplex* B, cuComplex* C, int Am, int An, int Ak, int Bn) {
    cublasHandle_t handle;
    cuComplex alpha;
    alpha.x =1;
    alpha.y =0;
    cuComplex beta;
    beta.x = 0;
    beta.y = 0;
    int Bm = An;
    int Bk = Ak;
    //int strA = Am*An;
    //int strB = Bm*Bn;
    //int strC = Am*Bn;
    //int batchCount = Ak;

    cuComplex *d_A, *d_B, *d_C;
    cudaMalloc ((void**)&d_A, sizeof(cuComplex) * Am*An*Ak);
    cudaMalloc ((void**)&d_B, sizeof(cuComplex) * Bm*Bn*Bk);
    cudaMalloc ((void**)&d_C, sizeof(cuComplex) * Am*Bn*Ak);

    cudaMemcpy(d_A, A, sizeof(cuComplex) * Am*An*Ak, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(cuComplex) * Bm*Bn*Bk, cudaMemcpyHostToDevice);

    for (int i=0; i<1; i++)
    {
    cublasCreate(&handle);
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am*Am, Bn*Am, Bm, &alpha, d_A, Am*Am, d_B, Bm, &beta, d_C, Am*Am);
    //cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A+strA*i, Am, d_B+strB*i, Bm, &beta, d_C+strC*i, Am);
    }
    cudaMemcpy(C, d_C, sizeof(cuComplex) * Am*Bn*Ak, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

