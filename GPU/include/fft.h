#ifndef GUARD_fft_h
#define GUARD_fft_h
#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#define MAX_PLAN1D_SIZE  256 
void basedTfft(float *t,int l,int bat,cufftComplex *tf);
void basedTifft(float *t, int l, int bat, cufftComplex *tf);
void streamedTfft(float *t,int l,int bat,cufftComplex *tf);
void streamedTifft(float *t, int l, int bat, cufftComplex *tf);
void batchedTfft(float *t,int l,int bat,cufftComplex *tf);
void batchedTifft(float *t,int l,int bat,cufftComplex *tf);
#endif
