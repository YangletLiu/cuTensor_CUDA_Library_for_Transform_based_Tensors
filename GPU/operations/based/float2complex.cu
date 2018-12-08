#include "based.h"
__global__ void float2cuComplex(float* d_in,const int length,cuComplex* d_out){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid < length){
	d_out[tid].x = d_in[tid];
	__syncthreads();
	d_out[tid].y = 0;
	}
}
