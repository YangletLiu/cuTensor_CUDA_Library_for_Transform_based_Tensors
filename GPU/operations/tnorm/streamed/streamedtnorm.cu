#include "norm.h"
#include "based.h"
/**
* normlization of tnesor t:
*       t = v * a;
* ||v|| = 1,it's mean <v,v> = e.
* INPUT: t is m×1×n
*
* OUTPUT: v*a  = t.
*/
void streamedtnorm(float* t, const int m, const int n, float* v, float* a){
	int ht = n/2+1;
	int bat = m;
	float* d_t;
	cufftComplex* d_fftData;
	cudaMalloc((void**)&d_t,sizeof(float)*bat*n);
	cudaMalloc((void**)&d_fftData,sizeof(cufftComplex)*m*ht);
	cudaMemcpy(d_t,t,sizeof(float)*bat*n,cudaMemcpyHostToDevice);	
	
	//tfft
	cufftHandle plan;
	int n_f[1] = {n};
	int in[1] = {n};
	int ou[1] = {ht}; 
	int stride_in = bat,dist_in = 1;
	int stride_ou = bat,dist_ou = 1;
	
	if(cufftPlanMany(&plan,1,n_f,in,stride_in,dist_in,ou,stride_ou,dist_ou,
				CUFFT_R2C,bat) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	} 
	if(cufftExecR2C(plan,d_t,(cufftComplex*)d_fftData)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	cudaFree(d_t);
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]cufftDestory faile!",__FUNCTION__,__LINE__);
		return;
	}
	//solve normlize
	int threads =512;
	int blocks = 1;	
	//set stream
	
	cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*PLAN1D_SIZE);	

	#pragma unroll
	for(int i=0;i<PLAN1D_SIZE;i++){
	   cudaStreamCreate(&stream[i]);
	}

	cuComplex *d_hv,*d_ha;
	cudaMalloc((void**)&d_hv,sizeof(cuComplex)*m*ht);
	cudaMalloc((void**)&d_ha,sizeof(cuComplex)*ht);
	
	int tube_num = ht/PLAN1D_SIZE;
	int tube_s = ht%PLAN1D_SIZE;
	if(tube_num > 0){
	 for(int j=0;j< tube_num;j++){
	   for(int i=0;i<PLAN1D_SIZE;i++){
	      d_normlize<<<blocks,threads,0,stream[i]>>>(d_fftData+i*m+j*m*PLAN1D_SIZE,m,d_hv+i*m+j*m*PLAN1D_SIZE,d_ha+i+j*PLAN1D_SIZE);
	   }
	   }
	
	for(int i=0;i<tube_s;i++){
	     d_normlize<<<blocks,threads,0,stream[i]>>>(d_fftData+i*m+tube_num*m*PLAN1D_SIZE,m,d_hv+i*m+tube_num*m*PLAN1D_SIZE,d_ha+i+tube_num*PLAN1D_SIZE);
	}
	}else{
	for(int i=0;i<tube_s;i++){
	     d_normlize<<<blocks,threads,0,stream[i]>>>(d_fftData+i*m,m,d_hv+i*m,d_ha+i);
	}
	}

	#pragma unroll
	for(int i=0;i<PLAN1D_SIZE;i++){
	   cudaStreamSynchronize(stream[i]);
	}

	#pragma unroll
	for(int i=0;i<PLAN1D_SIZE;i++){
	   cudaStreamDestroy(stream[i]);
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaFree(d_fftData);	
	//d_hv and d_ha take ifft 
//	int threads = 0;
//	int blocks = 0;	
	int num = 0;
	
	float *d_v,*d_a;
	cudaMalloc((void**)&d_v,sizeof(float)*m*n);
	cudaMalloc((void**)&d_a,sizeof(float)*n);
 
	cufftHandle iplan;
	in[0] = ht;
	ou[0] = n;

	if(cufftPlanMany(&iplan,1,n_f,in,stride_in,dist_in,ou,stride_ou,dist_ou,
				CUFFT_C2R,bat) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	} 
	if(cufftExecC2R(iplan,d_hv,d_v)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	num = m*n;
	if(num < 512){
	    threads = num;
	    blocks = 1;
	}else{
	    threads = 512;
	    blocks = ((num%512) == 0)?num/512:num/512+1;
	}

	fftResultProcess<<<blocks,threads>>>(d_v,num,n);
	
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaMemcpy(v,d_v,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
	
	cudaFree(d_hv);
	cudaFree(d_v);
	
	stride_in = 1;
	stride_ou = 1;
	dist_in = 1;
	dist_ou = 1;
	bat = 1;

	if(cufftPlanMany(&iplan,1,n_f,in,stride_in,dist_in,ou,stride_ou,dist_ou,
				CUFFT_C2R,bat) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cufftExecC2R(iplan,d_ha,d_a)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	if(cufftDestroy(iplan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]cufftDestory faile!",__FUNCTION__,__LINE__);
		return;
 	}	
	num = n;
	if(n < 512){
	    threads = num;
	    blocks = 1;
	}else{
	    threads = 512;
	    blocks = ((num%512) == 0)?num/512:num/512+1;
	}	

	fftResultProcess<<<blocks,threads>>>(d_a,num,n);

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaMemcpy(a,d_a,sizeof(float)*n,cudaMemcpyDeviceToHost);
	
	cudaFree(d_ha);	
	cudaFree(d_a);
}
