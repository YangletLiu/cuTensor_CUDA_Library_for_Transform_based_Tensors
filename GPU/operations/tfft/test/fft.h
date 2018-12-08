#ifndef GUARD_fft_h
#define GUARD_fft_h
#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
void Tfft(float *t,int l,int bat,cufftComplex *tf);
void Tifft(float *t,int l,int bat,cufftComplex *tf);
#endif
