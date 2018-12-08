#ifndef BASED_H_
#define BASED_H_
#include <cuda.h>
#include <cufft.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define PLAN1D_SIZE 16
void batcheddiagmat(float *t,const int m,const int batch,float* result);
void batchedctranspose(cuComplex* A,const int m,const int n,const int batch,cuComplex* T);
void batchedftranspose(float* A,const int m,const int n,const int batch,float* T);
__global__ void extractEvenNumU(float* d_k, cuComplex* d_U,const int m, const int n, const int batch);
__global__ void extractEvenNumS(float* d_s, float* ds_extract,const int m, const int n, const int batch);
__global__ void fftResultProcess(float* t,const int num,const int len);
__global__ void float2cuComplex(float* d_in,const int length,cuComplex* d_out);
__global__ void conMatrixK(cufftComplex* d_fftData, float* d_k,const int m, const int n,const int batch);
#endif 
