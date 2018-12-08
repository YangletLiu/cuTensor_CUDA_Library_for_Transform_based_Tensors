//#include "gemmStrideStreamed.h"
#include"tprod.h"

void gemmStrideStreamed(cuComplex *A, cuComplex *B, cuComplex *C, int m, int n, int k, int l)
{
    //Allocate device memory for A B C
    cuComplex *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m*n*k*sizeof(cuComplex));
    cudaMalloc((void**)&d_B, n*l*k*sizeof(cuComplex));
    cudaMalloc((void**)&d_C, m*l*k*sizeof(cuComplex));

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    //transfer A B to device memory
    cudaMemcpy(d_A, A, sizeof(cuComplex)  *m*n*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(cuComplex)  *n*l*k, cudaMemcpyHostToDevice);

    cudaStream_t *streams = (cudaStream_t *) malloc(k*sizeof(cudaStream_t));
    for (int i=0; i<k; i++)
        cudaStreamCreate(&streams[i]);
    cuComplex alpha;
    alpha.x =1;
    alpha.y =0;
    cuComplex beta;
    beta.x = 0;
    beta.y = 0;

    for (int i=0; i<k; i++)
    {
        cublasSetStream(handle, streams[i]);

        cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha,
                d_A+i*m*n, m, d_B+i*n*l, n, &beta, d_C+i*m*l, m);
    }

    cudaMemcpy(C, d_C, sizeof(cuComplex)*m*l*k, cudaMemcpyDeviceToHost);
    for (int i=0; i<k; i++)
        cudaStreamDestroy(streams[i]);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(streams);
    cublasDestroy(handle);
}
