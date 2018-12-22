#ifndef _SOL_V_H_
#define _SOL_V_H_
#include"based.h"
#include"fft.h"
/**
* solve tensor US*V=T
* US is a tensor , size: m * min(m,n) * k
* T is a tensor , size: m * n * k
* V is a tensor ,size: min(m,n) * n * k 
*/
void solve_v(float* T,float* US,const int m,const int n,const int k,float* V);
#endif
