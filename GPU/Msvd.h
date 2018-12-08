#ifndef MSVD_H
#define MSVD_H
#include <stdlib.h>
#include <stdio.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <assert.h>
void Msvd(cufftComplex *tp, cufftComplex *up, cufftComplex *vp, float *sp, int m, int n, int i);
#endif
