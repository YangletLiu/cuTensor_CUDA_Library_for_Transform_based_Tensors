#ifndef TPROD_H
#define TPROD_H
#include "fft.h"
#include <stdio.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
void gemmStrideBatched(cuComplex *A, cuComplex *B, cuComplex *C, int Am, int An, int Ak,  int Bn);
void gemmStrideStreamed(cuComplex *A, cuComplex *B, cuComplex *C, int m, int n, int k, int l);
void gemmStrideBased(cuComplex* A, cuComplex* B, cuComplex* C, int Am, int An, int Ak, int Bn);
void basedtprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe);
void streamedtprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe);
void batchedtprod(float* t1,float* t2,float* T,cublasOperation_t t_t1,cublasOperation_t t_t2,int row, int col, int rank, int tupe);
#endif

