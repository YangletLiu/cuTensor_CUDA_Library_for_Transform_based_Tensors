#include "svd.h"
#include "based.h"
void basedtsvd(float* t,const int m,const int n,const int tupe,float* U,float* S,float*  V){
	int bat = m*n;
	cufftComplex* t_f = (cufftComplex*)malloc(bat*tupe*sizeof(cufftComplex));
	//transform
	for(int i=0;i<bat;i++){
	   for(int j=0;j<tupe;j++){
		t_f[i*tupe+j].x=t[j*bat+i];
		t_f[i*tupe+j].y=0;
		}
	}

	//tfft:C2C
	cufftComplex* d_fftData;
	cudaMalloc((void**)&d_fftData,tupe*bat*sizeof(cufftComplex));	
	cudaMemcpy(d_fftData,t_f,bat*tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice);

	cufftHandle plan;
	if(cufftPlan1d(&plan,tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
	 	fprintf(stdout,"[%s]:[%d] cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
	}
	for(int i=0;i<bat;i++){
	if(cufftExecC2C(plan,d_fftData+i*tupe,d_fftData+i*tupe,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}

	//transform
	cudaMemcpy(t_f,d_fftData,sizeof(cufftComplex)*bat*tupe,cudaMemcpyDeviceToHost);
	cufftComplex* t_f2 = (cufftComplex*)malloc(sizeof(cufftComplex)*tupe*bat);

	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		t_f2[j*bat+i]=t_f[i*tupe+j];
		}
	}
	
/*printf("\n============================\n");
for(int i=0;i<bat*tupe;i++){
    printf("[%f %f]",t_f2[i].x,t_f2[i].y);
}	
printf("\n============================\n");
*/	cudaMemcpy(d_fftData,t_f2,sizeof(cufftComplex)*bat*tupe,cudaMemcpyHostToDevice);
	
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
	//tsvd
	cusolverDnHandle_t handle;
	gesvdjInfo_t params;
	int* info = NULL;
	int echo = 1;
	int lda = m;
	int ldu = m;
	int ldv = n;
	int lwork = 0;
	cuComplex* work=NULL;

	//malloc u s v

	float* d_s = NULL;
	cuComplex* d_u = NULL;
	cuComplex* d_v = NULL;
	cudaMalloc((void**)&d_s,sizeof(float)*tupe*((m<n)?m:n));
	cudaMalloc((void**)&d_u,sizeof(cuComplex)*tupe*m*((m<n)?m:n));
	cudaMalloc((void**)&d_v,sizeof(cuComplex)*tupe*n*((m<n)?m:n));
	cudaMalloc((void**)&info,sizeof(int));	
	
	if(cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnCreate failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	if(cusolverDnCreateGesvdjInfo(&params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:creation svd info srror",__FUNCTION__,__LINE__);
		return;
	}	
	
	if(cusolverDnCgesvdj_bufferSize(
			handle,
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_fftData,
			m,
			d_s,
			d_u,
			ldu,
			d_v,
			ldv,
			&lwork,
			params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR: create buffersize failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaMalloc((void**)&work,sizeof(cuComplex)*lwork);

	int step_d = m*n;
	int step_u = m*((m<n)?m:n);
	int step_s = ((m<n)?m:n);
	int step_v = n*((m<n)?m:n);	
	
	for(int i=0;i<tupe;i++){
	  if(cusolverDnCgesvdj(
			handle,
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_fftData+step_d*i,
			lda,
			d_s+i*step_s,
			d_u+i*step_u,
			ldu,
			d_v+i*step_v,
			ldv,
			work,
			lwork,
			info,
			params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:cusolverDnCgesvdj failed!",__FUNCTION__,__LINE__);
		return;
		}
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	if(cusolverDnDestroy(handle)!=CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cusolverDnDestroyGesvdjInfo(params)!=CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	if(d_fftData != NULL){
	cudaFree(d_fftData);
	d_fftData = NULL;
	}
	if(work != NULL){
	cudaFree(work);
	work = NULL;
	}
	if(info != NULL){
	cudaFree(info);
	info = NULL;
	}

	//ifft
	
	//transform
	cuComplex* h_u = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_u);
	cuComplex* h_u2 = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_u);
	cuComplex* h_v = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_v);
	cuComplex* h_v2 = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_v);
	cuComplex* h_s = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_s);
	float* h_s2 = (float*)malloc(sizeof(float)*tupe*step_s);
	cuComplex* d_s2;
	cudaMalloc((void**)&d_s2,sizeof(cuComplex)*tupe*step_s);

	cudaMemcpy(h_u2,d_u,sizeof(cuComplex)*tupe*step_u,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v2,d_v,sizeof(cuComplex)*tupe*step_v,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_s2,d_s,sizeof(float)*tupe*step_s,cudaMemcpyDeviceToHost);
/*printf("\n============================\n");
for(int i=0;i<tupe*step_s;i++){
    printf("[%f ]",h_s2[i]);
}	
printf("\n============================\n");
*/	
	//transform_u
	for(int i=0;i<step_u;i++){
	  for(int j=0;j<tupe;j++){
		h_u[i*tupe+j]=h_u2[j*step_u+i];
		}
	}

	//transform_v
	for(int i=0;i<step_v;i++){
	  for(int j=0;j<tupe;j++){
		h_v[i*tupe+j]=h_v2[j*step_v+i];
		}
	}
		
	//transform_s
	for(int i=0;i<step_s;i++){
	  for(int j=0;j<tupe;j++){
		h_s[i*tupe+j].x=h_s2[j*step_s+i];
		h_s[i*tupe+j].y=0;
		}
	}
	
/*	for(int i=0;i<tupe*step_s;i++){
		printf("%f ",h_s2[i]);
	}
	printf("\n");
*/	cudaMemcpy(d_u,h_u,sizeof(cuComplex)*tupe*step_u,cudaMemcpyHostToDevice);
	cudaMemcpy(d_s2,h_s,sizeof(cuComplex)*tupe*step_s,cudaMemcpyHostToDevice);
	cudaMemcpy(d_v,h_v,sizeof(cuComplex)*tupe*step_v,cudaMemcpyHostToDevice);
	
	if(h_u2 != NULL){
	free(h_u2);
	h_u2 = NULL;
	}
	if(h_v2 != NULL){
	free(h_v2);
	h_v2 = NULL;
	}
	if(h_s2 != NULL){
	free(h_s2);
	h_s2= NULL;
	}
	if(d_s != NULL){
	cudaFree(d_s);
	d_s = NULL;
	}

	cufftHandle iplan;

	if(cufftPlan1d(&iplan,tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;	
	}
	

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	//ifft_u
	for(int i=0;i<step_u;i++){
	if(cufftExecC2C(iplan,d_u+i*tupe,d_u+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
	

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	//ifft_v
	if(cufftPlan1d(&iplan,tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;	
	}
	for(int i=0;i<step_v;i++){
	if(cufftExecC2C(iplan,d_v+i*tupe,d_v+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]CUFFT ERROR: cufftExecc2Cfailed!",__FUNCTION__,__LINE__);
		return;
	}
	}

	

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	//ifft_s
	if(cufftPlan1d(&iplan,tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;	
	}
	for(int i=0;i<step_s;i++){
	if(cufftExecC2C(iplan,d_s2+i*tupe,d_s2+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	
	}
	}
	
	//transform
		
	cudaMemcpy(h_u,d_u,sizeof(cuComplex)*tupe*step_u,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v,d_v,sizeof(cuComplex)*tupe*step_v,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_s,d_s2,sizeof(cuComplex)*tupe*step_s,cudaMemcpyDeviceToHost);

	//transform_u
	for(int i=0;i<step_u;i++){
	  for(int j=0;j<tupe;j++){
		U[j*step_u+i]=h_u[i*tupe+j].x/tupe;
	//	U[j*step_u+i].y=h_u[i*tupe+j].y/tupe;
		}
	}

	//transform_v
	for(int i=0;i<step_v;i++){
	  for(int j=0;j<tupe;j++){
		V[j*step_v+i]=h_v[i*tupe+j].x/tupe;
	//	V[j*step_v+i].y=h_v[i*tupe+j].y/tupe;
		}
	}	

	//transform_s
	for(int i=0;i<step_s;i++){
	  for(int j=0;j<tupe;j++){
		S[j*step_s+i]=h_s[i*tupe+j].x/tupe;
	//	S[j*step_s+j].y=h_s[i*tupe+j].y/tupe;
		}
	}
	
	if(cufftDestroy(iplan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(d_u != NULL){	
	cudaFree(d_u);
	d_u =NULL;
	}
	if(d_v != NULL){
	cudaFree(d_v);
	d_v = NULL;
	}
	if(d_s2 != NULL){
	cudaFree(d_s2);
	d_s2 = NULL;
	}
	if(h_u !=NULL){
	free(h_u);
	h_u = NULL;
	}
	if(h_v != NULL){
	free(h_v);
	h_v = NULL;
	}
	if(h_s != NULL){
	free(h_s);
	h_s = NULL;
	}
}
