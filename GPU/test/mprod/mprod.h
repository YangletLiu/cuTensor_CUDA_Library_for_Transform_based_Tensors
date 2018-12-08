#ifndef MPROD_H_
#define MPROD_H_
#include <stdio.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>

void mprod(cuComplex* A,cuComplex* B,cuComplex* C,int Am,int An,int Bn);
#endif
