#ifndef BATCHED_H_
#define BATCHED_H_
#include<stdlib.h>
#include"based.h"
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cufft.h>
#include"magma_v2.h"
#include"magma_lapack.h"
#define min(a,b) (((a)<(b))?(a):(b))
void batchedtqr(float* t,const int m,const int n,const int tupe,cuComplex* tau);
#endif

