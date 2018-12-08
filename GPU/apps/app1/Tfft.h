#ifndef GUARD_fft_h
#define GUARD_fft_h
#include <cuda.h>
#include <cufft.h>
#include "head.h"
void Tfft(float *t,int l,int bat,cufftComplex *tf);
void cufft(float *t,int l,int bat,cufftComplex *tf);
void printTensor(int m, int n,int k, const float*A);
void TprintTensor(int m, int n,int k, const cufftComplex *A);
void Tifft(float *t,int l,int bat,cufftComplex *tf);
void mul_cufft(cufftComplex *a,cufftComplex *b,cufftComplex *c);
void transform(int a,int b,int c,cufftComplex *t,cufftComplex *tt);
double cpuSecond();
#endif
