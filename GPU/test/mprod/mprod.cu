#include "mprod.h"
void mprod(cuComplex* A, cuComplex* B, cuComplex* C, int Am, int An, int Bn) {
    cublasHandle_t handle;
    cuComplex alpha;
    alpha.x =1;
    alpha.y =0;
    cuComplex beta;
    beta.x = 0;
    beta.y = 0;
    int Bm = An;

    cuComplex *d_A, *d_B, *d_C;
    cudaMalloc ((void**)&d_A, sizeof(cuComplex) * Am*An);
    cudaMalloc ((void**)&d_B, sizeof(cuComplex) * Bm*Bn);
    cudaMalloc ((void**)&d_C, sizeof(cuComplex) * Am*Bn);

    cudaMemcpy(d_A, A, sizeof(cuComplex) * Am*An, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(cuComplex) * Bm*Bn, cudaMemcpyHostToDevice);

   if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
	fprintf(stdout,"[%s][%d] cublasCreate faile!",__FUNCTION__,__LINE__);
	return;
	}
    if(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, Am, d_B, Bm, &beta, d_C, Am) !=CUBLAS_STATUS_SUCCESS){
	fprintf(stdout,"[%s]:[%d] cublasCgemm faile!",__FUNCTION__,__LINE__);
	return;
	
	}
    
    cudaMemcpy(C, d_C, sizeof(cuComplex) * Am*Bn, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

