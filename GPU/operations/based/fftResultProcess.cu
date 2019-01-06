#include "based.h"
#if 1
__global__ void fftResultProcess(float* d_t,const int num,const int len){
const int tid = blockIdx.x*blockDim.x+threadIdx.x;
if(tid < num){
	d_t[tid]=d_t[tid]/len;
	}
	__syncthreads();
}

__global__ void fftResultProcess(cuComplex* d_t,const int num, const int len){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid < num){
	d_t[tid].x=d_t[tid].x/len;
	__syncthreads();
	d_t[tid].y=d_t[tid].y/len; 
	}
	__syncthreads();	
}

#endif

#if 0
__global__ void d_frp(float* d_t,const int num,const int len){
const int tid = blockIdx.x*blockDim.x+threadIdx.x;
if(tid < num){
	d_t[tid]=d_t[tid]/len;
	}
	__syncthreads();
}

__global__ void d_frp(cuComplex* d_t,const int num, const int len){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid < num){
	d_t[tid].x=d_t[tid].x/len; 
	d_t[tid].y=d_t[tid].y/len;
	}
	__syncthreads();	
}


void fffResultProcess(float* d_t,const int num,const int len){
	int threads = 0;
	int blocks = 0;
	if( num < 512){
	   threads = num;
	   blocks = 1;	
	}else{
	   threads = 512;
	   blocks = ((num%512) == 0)?num/512:num/512+1;	
	}
	d_frp<<<blocks,threads>>>(d_t,num,len);
}

void fffResultProcess(cuComplex* d_t,const int num,const int len){
	int threads = 0;
	int blocks = 0;
	if( num < 512){
	   threads = num;
	   blocks = 1;	
	}else{
	   threads = 512;
	   blocks = ((num%512) == 0)?num/512:num/512+1;	
	}
	d_frp<<<blocks,threads>>>(d_t,num,len);
}
#endif
