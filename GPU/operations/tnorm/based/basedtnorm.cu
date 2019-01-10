#include "norm.h"
#include "based.h"
void basedtnorm(float* t,const int m,const int n,float* v,float* a){
	int bat = m;
	int tube = n;
	cufftComplex* t_f = (cufftComplex*)malloc(sizeof(cufftComplex)*m*n);
	//transform forward
	for(int i=0;i<bat;i++){
	  for(int j=0;j<tube;j++){
		t_f[i*tube+j].x = t[j*bat+i];
		t_f[i*tube+j].y = 0;
	  }
	}

	//tfft:C2C
	cufftComplex* d_fftData;
	cudaMalloc((void**)&d_fftData,sizeof(cufftComplex)*m*n);
	cudaMemcpy(d_fftData,t_f,sizeof(cuComplex)*m*n,cudaMemcpyHostToDevice);
	
	cufftHandle plan;
	if(cufftPlan1d(&plan,tube,CUFFT_C2C,1) != CUFFT_SUCCESS){
	 	fprintf(stdout,"[%s]:[%d] cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
	}
	for(int i=0;i<bat;i++){
	if(cufftExecC2C(plan,d_fftData+i*tube,d_fftData+i*tube,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
	
	//transform inverse

	cudaMemcpy(t_f,d_fftData,sizeof(cuComplex)*m*n,cudaMemcpyDeviceToHost);
	cufftComplex* t_f2 = (cufftComplex*)malloc(sizeof(cuComplex)*m*n);
	
	for(int i=0;i<bat;i++){
	  for(int j=0;j<tube;j++){
		t_f2[j*bat+i]=t_f[i*tube+j];
	  }
	}
	
	cudaMemcpy(d_fftData,t_f2,sizeof(cuComplex)*m*n,cudaMemcpyHostToDevice);

	if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
		
	if(t_f != NULL){
	free(t_f);
	t_f = NULL;
	}

	if(t_f2 !=NULL){
	free(t_f2);
	t_f2 = NULL;	
	}
	//solve normlize
	cuComplex *d_v,*d_a;
	cudaMalloc((void**)&d_v,sizeof(cuComplex)*m*n);
	cudaMalloc((void**)&d_a,sizeof(cuComplex)*n);
	
	#pragma unroll
	for(int i=0;i<n;i++){

	  normlize(d_fftData+i*m,m,1,d_v+i*m,d_a+i);

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
		}
	  
	}

	cudaFree(d_fftData);
	
	//transform forward
	cuComplex* h_v = (cuComplex*)malloc(sizeof(cuComplex)*m*n);
	cuComplex* h_v2 = (cuComplex*)malloc(sizeof(cuComplex)*m*n);
	cudaMemcpy(h_v2,d_v,sizeof(cuComplex)*m*n,cudaMemcpyDeviceToHost);
	for(int i=0;i<bat;i++){
	  for(int j=0;j<tube;j++){
		h_v[i*tube+j] = h_v2[j*bat+i];
	  }
	}
	cudaMemcpy(d_v,h_v,sizeof(cuComplex)*m*n,cudaMemcpyHostToDevice);

	if(h_v2){
	   free(h_v2);
	   h_v2 = NULL;
	}
	//d_v and d_a take ifft
	cufftHandle iplan;
	if(cufftPlan1d(&iplan,tube,CUFFT_C2C,1) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;	
	}

	for(int i=0;i<bat;i++){
	if(cufftExecC2C(iplan,d_v+i*tube,d_v+i*tube,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
	

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	//d_a take ifft
	if(cufftPlan1d(&iplan,tube,CUFFT_C2C,1) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;	
	}
	if(cufftExecC2C(iplan,d_a,d_a,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	cuComplex* h_a = (cuComplex*)malloc(sizeof(cuComplex)*n);
	cudaMemcpy(h_v,d_v,sizeof(cuComplex)*m*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_a,d_a,sizeof(cuComplex)*n,cudaMemcpyDeviceToHost);
	//transform inverse
	for(int i=0;i<bat;i++){
	   for(int j=0;j<tube;j++){
		v[j*bat+i]=h_v[i*tube+j].x/tube;
	   }
	}
	
	for(int i=0;i<tube;i++){
	  a[i]=h_a[i].x/tube;
	}
	
	if(cufftDestroy(iplan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(h_v != NULL){
	   free(h_v);
	   h_v = NULL;
	}
	if(h_a != NULL){
	   free(h_a);
	   h_a = NULL;
	}
	if(d_v != NULL){
	   cudaFree(d_v);
	   d_v = NULL;
	}
	if(d_a != NULL){
	   cudaFree(d_a);
	   d_a = NULL;
	}
}
