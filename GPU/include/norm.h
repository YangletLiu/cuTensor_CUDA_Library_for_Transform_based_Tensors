#ifndef _NORM_
#define _NORM_
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "stdio.h"
void batchedtnorm(float* t,const int m,const int n,const int tube,float* result);
#endif
