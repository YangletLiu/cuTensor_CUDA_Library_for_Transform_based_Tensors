#ifndef SVD_H_
#define SVD_H_
#include <cufft.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "kblas.h"
#include "batch_svd.h"
void basedtsvd(float* t,const int m,const int n,const int tupe,float* U,float* S,float* V);
void streamedtsvd(float* t,const int m,const int n,const int tupe,float* U,float* S,float* V);
void batchedtsvd(float* t,const int m,const int n, const int tupe, float* host_u,float* S);
#endif

