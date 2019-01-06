#include"based.h"
/**
*INPUT: d_hu of size m*n*(batch/2+1).
*OUTPUT: d_u of size m*n*batch.
*batch is length of d_u
*/
__global__ void  d_symmetricRecoverU(cuComplex* d_hu,const int m,const int n,const int batch,cuComplex* d_u){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	int ht = batch/2+1;
	int num = m*n*ht;
	if( tid < num){
	
	int id_mn = tid/(m*n);
	int ids_mn = tid%(m*n);

	d_u[ids_mn + id_mn*m*n] = d_hu[ids_mn+ id_mn*m*n];	
//	d_u[tid] = d_hu[tid];
	__syncthreads();

	if( (batch%2) == 1){
	   if( 0 < id_mn < ht){
		d_u[ (ht-id_mn-1)*m*n +ids_mn + ht*m*n].x=d_hu[ id_mn*m*n + ids_mn].x;
		__syncthreads();
		d_u[ (ht-id_mn-1)*m*n +ids_mn + ht*m*n].y=0-d_hu[ id_mn*m*n + ids_mn].y;
		}
	}
	__syncthreads();
	
	if( (batch%2) == 0){
		if( 0 < id_mn < (ht-1)){
		d_u[(ht-id_mn-2)*m*n +ids_mn + ht*m*n].x=d_hu[ id_mn*m*n + ids_mn ].x;
		__syncthreads();
		d_u[(ht-id_mn-2)*m*n +ids_mn + ht*m*n].y=0-d_hu[ id_mn*m*n + ids_mn ].y;
		}
	}
	__syncthreads();

	}
}

void symmetricRecoverU(cuComplex* d_hu,const int m,const int n,const int batch,cuComplex* d_u){
	int threads = 0;
	int blocks = 0;
	int ht = batch/2+1;
	int num =m*n*ht;
	if( num < 512){
	   threads = num;
	   blocks = 1;
	}else{
	   threads = 512;
	   blocks = ((num%512) == 0)?num/512:num/512+1;
	}
	d_symmetricRecoverU<<<blocks,threads>>>(d_hu,m,n,batch,d_u);
}
