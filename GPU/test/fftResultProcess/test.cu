#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include"based.h"
/*__global__ void fftResultProcess(float* d_t,const int num,const int len){
const int tid = blockIdx.x*blockDim.x+threadIdx.x;
if(tid < num){
	d_t[tid]=d_t[tid]/len;
	}
	__syncthreads();
}*/
int main(){
	int num=100;
	int len=2;
	float* data = new float[num];
	float* out = new float[num];
	for(int i=0;i<num;i++){
	data[i]=i;	
	} 
	float* d_data;
	cudaMalloc((void**)&d_data,sizeof(float)*num);
	cudaMemcpy(d_data,data,sizeof(float)*num,cudaMemcpyHostToDevice);
	int threads=0;
	int blocks=0;
	if(num<512){
	threads=num;
	blocks=1;
	}else{
	threads=512;
	blocks=(num%512 ==0)?num/512:num/512+1;
	}
	fftResultProcess<<<blocks,threads>>>(d_data,num,len);
	cudaMemcpy(out,d_data,sizeof(float)*num,cudaMemcpyDeviceToHost);
	for(int i=0;i<num;i++){
	std::cout<<out[i]<<std::endl;
	}
	return 0;
	}

