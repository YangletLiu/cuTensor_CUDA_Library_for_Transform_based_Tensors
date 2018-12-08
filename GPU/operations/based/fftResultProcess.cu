#include "based.h"
__global__ void fftResultProcess(float* d_t,const int num,const int len){
const int tid = blockIdx.x*blockDim.x+threadIdx.x;
if(tid < num){
	d_t[tid]=d_t[tid]/len;
	}
	__syncthreads();
}
