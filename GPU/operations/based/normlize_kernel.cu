#include "based.h"
__global__ void d_normlize(cuComplex* d_in,const int len,cuComplex* v,cuComplex* a){
	int tidx = threadIdx.x;
	const int bidx = blockIdx.x;
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	int t_n = blockDim.x;
	__shared__ float temp_real[512];
	__shared__ float temp_image[512];
	float t_real = 0.0;
	float t_image = 0.0;
	while(tidx < len){
		t_real += d_in[tidx+bidx*len].x * d_in[tidx+bidx*len].x;
		t_image += d_in[tidx+bidx*len].y * d_in[tidx+bidx*len].y;
		 tidx += t_n;
	} 
	tidx = threadIdx.x;	
	temp_real[tidx] = t_real;
	temp_image[tidx] = t_image;
	__syncthreads();
	
	int i = 512/2;
	while(i != 0){
		if(tidx < i){
		  temp_real[tidx] += temp_real[tidx+i];
		  temp_image[tidx] += temp_image[tidx+i];
		}
		i /= 2;
	}
	if(tidx == 0){
	   temp_real[0]+=temp_image[0];
	}
	__syncthreads();
	
	if(temp_real[0] < 1.e-100){
		if(tidx == 0)
		{
		temp_real[0]= 2*len;
		}
	__syncthreads();
		while(tidx < len){
		   d_in[ tidx + bidx*len ].x = 1;
		   d_in[ tidx + bidx*len ].y = 1;
		   tidx += t_n;
		}
		tidx = threadIdx.x;
	__syncthreads();
	}
	
	if( tidx == 0){
		a[bidx].x = sqrt(temp_real[0]);
		a[bidx].y = 0;
	}
	__syncthreads();
	
	while( tidx < len){
		v[tidx + bidx*len].x = d_in[tidx + bidx*len].x/a[bidx].x;
		v[tidx + bidx*len].y = d_in[tidx + bidx*len].y/a[bidx].x;
		tidx += t_n;
	}
}

void normlize(cuComplex* d_in,const int len,const int batch,cuComplex* d_v,cuComplex* d_a){
	int threads = 512;
	int blocks = batch;
	d_normlize<<<blocks,threads>>>(d_in,len,d_v,d_a);
}
