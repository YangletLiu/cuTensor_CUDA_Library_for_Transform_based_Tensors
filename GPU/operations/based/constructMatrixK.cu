#include "based.h"
__global__ void conMatrixK(cufftComplex* d_fftData, float* d_k,const int m, const int n,const int batch){
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int num = m*n;
	int id;
	int tid;

	if(tidx < m*n*batch*4){

	id = tidx/(m*n*4);
	tid = tidx%(m*n*4);

	if(tid < num){
		d_k[tid/m*2*m+tid%m +id*4*m*n] = d_fftData[tid+id*m*n].x;
	}
	
	if(num <= tid < 2*num){
		d_k[(tid%num)/m*2*m+(tid%num)%m+m+id*4*m*n] = -d_fftData[tid%num+id*m*n].y;
	}

	if(2*num <= tid < 3*num){
		d_k[(tid%num)/m*2*m+(tid%num)%m+2*m*n +id*4*m*n] = d_fftData[tid%num+id*m*n].y;
		
	}
	
	if(3*num <= tid < 4*num){
		d_k[(tid%num)/m*2*m+(tid%num)%m+m+2*m*n +id*4*m*n] = d_fftData[tid%num+id*m*n].x;
		
	}

	}
	__syncthreads();
}
