#include "based.h"
/**
* note the array of result need to set 0;
*/
__global__ void d_batch_diag_mat(float* t,const int m,const int batch ,float* result){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int t_n=gridDim.x*blockDim.x;
    while(tid < m*m*batch){
	result[tid]=0;
	tid +=t_n;
    }
    __syncthreads();

    tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid < m*batch){
        result[tid/m*(m*m)+tid%m*(m+1)]=t[tid];
    }
    __syncthreads();
}

void batcheddiagmat(float *t,const int m,const int batch,float* result){
    int threads;
    int blocks;
    int num = m*batch;
    if(num < 512){
        threads = num;
        blocks = 1;
    }else{
        threads = 512;
        blocks = (num%512 == 0)?num/512:num/512+1;
    }

    d_batch_diag_mat<<<blocks,threads>>>(t,m,batch,result);
}
