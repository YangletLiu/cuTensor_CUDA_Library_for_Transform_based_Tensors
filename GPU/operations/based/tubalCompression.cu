#include "based.h"
/**
  t:the data of input.
  length:  each signal lengeth.
  batch:   the number of signal
  k :   the ratio of tubal.
*/
__global__ void d_tubal_compression(float* t,const int length,const int batch,const int k){
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int l = length - k;
	int num = k*batch;
	if( tid < num ){
	    int tid_l = tid/k;
	    int tid_l_s = tid%k;
	    t[tid_l*length+l+tid_l_s]=0;
	}
	__syncthreads();
}

void tubalCompression(float* t,const int length,const int batch, const int k){
	int threads = 0;
	int blocks = 0;
	int num = k*batch;
	if(num < 512){
	    threads = num;
	    blocks = 1;
	}else{
	    threads = 512;
	    blocks = ((num%512) == 0)?num/512:num/512+1;
	}
	
	d_tubal_compression<<<blocks,threads>>>(t,length,batch,k);
}
