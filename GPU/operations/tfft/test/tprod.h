#ifndef TPROD_H
#define TPROD_H
#include "fft.h"
#include <cublas_v2.h>
//#include "gemmStrideBatched.h"
//#include "gemmStrideStreamed.h"

void tprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe);
//void streamedtprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe);
#endif

