#include"based.h"
__global__ void extractEvenNumU(float* d_k, cuComplex* d_U, const int m,const int n, const int batch){
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int id,sid;
	int tid,stid;
	
	if(tidx< 2*m*n*batch){

	id = tidx/(2*m*n);	
	tid = tidx%(2*m*n);

	sid = tid/(2*m);
	stid = tid%(2*m);
	if(stid < m){
		d_U[stid + sid*m +id*m*n].y=d_k[ stid + sid*4*m + id*m*n*4];
	}

	if(m <= stid < 2*m){
		d_U[stid%m + sid*m +id*m*n].x=d_k[ stid%m + m +sid*4*m + id*m*n*4];
	}
	}	
	__syncthreads();
}
__global__ void extractEvenNumS(float* d_s, float* ds_extract, const int m,const int n, const int batch){
	const int tidx = blockIdx.x * blockDim.x +threadIdx.x;
	int min = ((2*m<2*n)?2*m:2*n);
	int id;
	int tid;
	if(tidx < batch*(min/2)){
		id = tidx/(min/2);
		tid = tidx%(min/2);
		ds_extract[tid + id*(min/2)]=d_s[tid*2 + id*min];
	}	
	__syncthreads();
}
