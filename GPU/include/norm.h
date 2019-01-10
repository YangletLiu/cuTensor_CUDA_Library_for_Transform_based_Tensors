#ifndef _NORM_
#define _NORM_
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "stdio.h"
void basedtnorm(float* t ,const int m,const int n,float* v, float* a);
void streamedtnorm(float* t ,const int m,const int n,float* v, float* a);
void batchedtnorm(float* t,const int m,const int n,float* v,float* a);
#endif
