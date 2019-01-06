#include "based.h"

__global__ void float2cuComplex(const float* d_in,const int length,cuComplex* d_out){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid < length){
	d_out[tid].x = d_in[tid];
	__syncthreads();
	d_out[tid].y = 0;
	}
}

__global__ void cuComplex2float(const cuComplex* d_in,const int length,float* d_out){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid < length){
	d_out[tid] = d_in[tid].x;
	}
}
#if 0
__global__ void d_float2cuComplex(const float* d_in,const int length,cuComplex* d_out){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid < length){
	d_out[tid].x = d_in[tid];
	__syncthreads();
	d_out[tid].y = 0;
	}
}

__global__ void d_cuComplex2float(const cuComplex* d_in,const int length,float* d_out){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid < length){
	d_out[tid] = d_in[tid].x;
	}
}
void float2Complex(const float* d_in, const int length,cuComplex* d_out){
	int threads = 0;
	int blocks = 0;
	if(length < 512){
`	    threads = length;
	    blocks =  1;
	}else{
	    threads = 512;
	    blocks = ((length%512) == 0)?length/512:length/512+1;
	}
	d_float2cuComplex<<<blocks,threads>>>(d_in,length,d_out);
}

void cuComplex2float(const cuComplex* d_in,const int length,float* d_out){
	int threads = 0;
	int blocks = 0;
	if(length < 512){
	   threads = length;
	   blocks = 1;
	}else{
	   threads = 512;
	   blocks = ((length%512) == 0)?length/512:length/512+1;
	}
	d_cuComplex2float<<<blocks,threads>>>(d_in,length,d_out);
}
#endif
