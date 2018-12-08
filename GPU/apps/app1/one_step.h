#ifndef GUARD_one_step_h
#define GUARD_one_step_h
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cufft.h>
#include "Tfft.h"
#include "head.h"
#include <iostream>
#include <sys/time.h> 
void one_step(cufftComplex* T_omega_f, cufftComplex* omega_f, cufftComplex* X_f, cufftComplex* Y_f, int m, int n, int k,int r_);
#endif
