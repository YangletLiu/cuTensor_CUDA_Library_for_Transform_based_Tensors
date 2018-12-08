#ifndef QR_H_
#define QR_H_
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cufft.h>
#define imin(x,y) ((x) < (y)) ? (x) : (y)
#define BILLION 1000000000L
void basedtqr(float* A,const int m,const int n,const int tupe,cuComplex* Tau);
void streamedtqr(float* A,const int m,const int n,const int tupe,cuComplex* Tau);
#endif
