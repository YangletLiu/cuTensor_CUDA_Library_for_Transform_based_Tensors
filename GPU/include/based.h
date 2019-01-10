#ifndef BASED_H_
#define BASED_H_
#include <cuda.h>
#include <cufft.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include"cusolverDn.h"
#include <stdio.h>
#include <math.h>
#define Min(a,b) ((a)<(b))?(a):(b)
#define PLAN1D_SIZE 500
/**
* normlization of tnesor t:
*       t = v * a;
* ||v|| = 1,it's mean <v,v> = e.
* INPUT: d_in is m×1×n
*
* OUTPUT: d_v * d_a  = d_in.
* len = n
* batch = m
*/
void normlize(cuComplex* d_in,const int len,const int batch,cuComplex* d_v,cuComplex* d_a);
__global__ void d_normlize(cuComplex* d_in,const int len,cuComplex* v,cuComplex* a);
/**
 INPUT/OUTPUT: t
  length:  each signal lengeth.
  batch:   the number of signal
  k :   the ratio of tubal.
*/
void tubalCompression(float* t,const int length,const int batch, const int k);
/**
*qrsolve is using to solve AX=B systems.
* A is a matrix,the size is m*n
* B is a matrix,the size is m*k
* X is a matrix,the size is min(m,n)*k
*/
void qrsolve(cuComplex* d_A,cuComplex* d_B,const int m,const int n,const int k,cuComplex* d_X);
/**
*INPUT: d_hu of size m*n*(batch/2+1).
*OUTPUT: d_u of size m*n*batch.
*batch is length of d_u
*/
void symmetricRecoverU(cuComplex* d_hu,const int m,const int n,const int batch,cuComplex* d_u);
/**
*INPUT: t is a array ,the size of m*batch,each vector size of m.
*OUTPT: the array of result
*/
void batcheddiagmat(float *t,const int m,const int batch,float* result);
/**
*/
void batchedctranspose(cuComplex* A,const int m,const int n,const int batch,cuComplex* T);
void batchedftranspose(float* A,const int m,const int n,const int batch,float* T);
/**
*/
__global__ void extractEvenNumU(float* d_k, cuComplex* d_U,const int m, const int n, const int batch);
__global__ void extractEvenNumS(float* d_s, float* ds_extract,const int m, const int n, const int batch);
/**
*INPUT: t
*OUTPUT: t
*the length of t is num,the len denote the size of signal. 
*/
__global__ void fftResultProcess(float* t,const int num,const int len);
__global__ void fftResultProcess(cuComplex* t,const int num,const int len);
#if 0
void fftResultProcess(float* t,const int num,const int len);
void fftResultProcess(cuComplex* t,const int num,const int len);
#endif
__global__ void float2cuComplex(const float* d_in,const int length,cuComplex* d_out);
__global__ void cuComplex2float(const cuComplex* d_in,const int length,float* d_out);
__global__ void conMatrixK(cufftComplex* d_fftData, float* d_k,const int m, const int n,const int batch);
#endif 
