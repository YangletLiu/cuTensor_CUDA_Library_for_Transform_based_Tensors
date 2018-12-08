#ifndef INV_H_
#define INV_H_

#include<stdio.h>
#include<stdlib.h>
#include<cufft.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include"magma_v2.h"
#include"based.h"
void basedtinv( float* t, const int m, const int n, const int tupe, float* invA);
void streamedtinv( float* t, const int m, const int n, const int tupe, float* invA);
void batchedtinv( float* t, const int m, const int n, const int tupe, float* invA);
#endif
