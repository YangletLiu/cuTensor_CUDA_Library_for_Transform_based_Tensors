/**
 * @device matview_transopse
 * Create on:Nov 28 2018
 * @author: haili
 * the size of tensor is m×n×batch
 */
#include "based.h"
__global__ void d_batch_c_transpose(cuComplex* A,const int m,const int n,const int batch,cuComplex* T){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int t_n=blockDim.x*gridDim.x;
	while(tid<m*n*batch){
		T[(tid/(m*n))*n*m+(tid%(m*n))/n+((tid%(m*n))%n)*m].x=A[tid].x;
		T[(tid/(m*n))*n*m+(tid%(m*n))/n+((tid%(m*n))%n)*m].y=0-A[tid].y;
		tid+=t_n;
	}
	__syncthreads();
}
__global__ void d_batch_f_transpose(float* A,const int m,const int n,const int batch,float* T){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int t_n=blockDim.x*gridDim.x;
	while(tid<m*n*batch){
		T[(tid/(m*n))*n*m+(tid%(m*n))/n+((tid%(m*n))%n)*m]=A[tid];
		tid+=t_n;
	}
	__syncthreads();
}

void batchedctranspose(cuComplex* A,const int m,const int n,const int batch,cuComplex* T){
    int threads;
    int blocks;
    int num= m * n * batch; 
    if(num < 512){
        threads=num;
        blocks=1;
    }else{
        threads=512;
        blocks= (num%512 ==0)?num/512:num/512+1;
    }

    d_batch_c_transpose<<<blocks,threads>>>(A,m,n,batch,T);
}
void batchedftranspose(float* A,const int m,const int n,const int batch,float* T){
    int threads;
    int blocks;
    int num= m * n * batch; 
    if(num < 512){
        threads=num;
        blocks=1;
    }else{
        threads=512;
        blocks= (num%512 ==0)?num/512:num/512+1;
    }

    d_batch_f_transpose<<<blocks,threads>>>(A,m,n,batch,T);
}
