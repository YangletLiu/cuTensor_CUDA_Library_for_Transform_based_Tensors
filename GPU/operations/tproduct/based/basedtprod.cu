#include "tprod.h"
#include "based.h"
void basedtprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe) {
	int bat = row*rank;
	int bat2 = rank*col;
	cufftComplex* t_f = (cufftComplex*)malloc(bat*tupe*sizeof(cufftComplex));
	cufftComplex* t_f2 = (cufftComplex*)malloc(bat2*tupe*sizeof(cufftComplex));
	//transform t1
	for(int i=0;i<bat;i++){
	   for(int j=0;j<tupe;j++){
		t_f[i*tupe+j].x=t1[j*bat+i];
		t_f[i*tupe+j].y=0;
		}
	}

	//transform t2
	for(int i=0;i<bat2;i++){
	   for(int j=0;j<tupe;j++){
		t_f2[i*tupe+j].x=t2[j*bat2+i];
		t_f2[i*tupe+j].y=0;
		}
	}

	//tfft:C2C
	cufftComplex* d_fftData;
	cufftComplex* d_fftData2;
	cudaMalloc((void**)&d_fftData,tupe*bat*sizeof(cufftComplex));	
	cudaMalloc((void**)&d_fftData2,tupe*bat2*sizeof(cufftComplex));	
	cudaMemcpy(d_fftData,t_f,bat*tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice);
	cudaMemcpy(d_fftData2,t_f2,bat2*tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftHandle plan2;

	if(cufftPlan1d(&plan,tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
	 	fprintf(stdout,"[%s]:[%d] cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	if(cufftPlan1d(&plan2,tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
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

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
	}
	for(int i=0;i<bat2;i++){
	if(cufftExecC2C(plan2,d_fftData2+i*tupe,d_fftData2+i*tupe,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
	}
	//transform
	cudaMemcpy(t_f,d_fftData,sizeof(cufftComplex)*bat*tupe,cudaMemcpyDeviceToHost);
	cudaMemcpy(t_f2,d_fftData2,sizeof(cufftComplex)*bat2*tupe,cudaMemcpyDeviceToHost);
	cufftComplex* t_f3 = (cufftComplex*)malloc(sizeof(cufftComplex)*tupe*bat);
	cufftComplex* t_f4 = (cufftComplex*)malloc(sizeof(cufftComplex)*tupe*bat2);

	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		t_f3[j*bat+i]=t_f[i*tupe+j];
		}
	}
	for(int i=0;i<bat2;i++){
	  for(int j=0;j<tupe;j++){
		t_f4[j*bat2+i]=t_f2[i*tupe+j];
		}
	}
	
	cudaMemcpy(d_fftData,t_f3,sizeof(cufftComplex)*bat*tupe,cudaMemcpyHostToDevice);
	cudaMemcpy(d_fftData2,t_f4,sizeof(cufftComplex)*bat2*tupe,cudaMemcpyHostToDevice);
	
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cufftDestroy(plan2)!=CUFFT_SUCCESS){
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
	if(t_f3 != NULL){
	free(t_f3);
	t_f3 = NULL;
	}
	if(t_f4 != NULL){
	free(t_f4);
	t_f4 = NULL;
	}
	//gemmbatched

	cufftComplex* d_Tf;
 	cudaMalloc((void**)&d_Tf,tupe*row*col*sizeof(cufftComplex));
	cublasHandle_t handle;
	cuComplex alpha;
	alpha.x =1;
	alpha.y =0;
	cuComplex beta;
	beta.x = 0;
	beta.y = 0;
	int Am = row;
	int An = rank;
	int Bn = col;
	int Bm = rank;
	int strA = Am*An;
	int strB = Bm*Bn;
	int strC = Am*Bn;
	cublasCreate(&handle);
	for(int i=0; i<tupe; i++){
	if(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
	        &alpha, d_fftData+i*strA, Am,d_fftData2+i*strB, Bm,  &beta,
	        d_Tf+i*strC, Am) !=CUBLAS_STATUS_SUCCESS){
	
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cublasCgemm failed!",__FUNCTION__,__LINE__);
		return;
	}
	  }
	cublasDestroy(handle);
		
	cudaFree(d_fftData);
	cudaFree(d_fftData2);
	//Tifft

	cuComplex* host_result=(cuComplex*)malloc(sizeof(cuComplex)*tupe*row*col);
	cuComplex* host_result2=(cuComplex*)malloc(sizeof(cuComplex)*tupe*row*col);

	cudaMemcpy(host_result,d_Tf,sizeof(cuComplex)*tupe*row*col,cudaMemcpyDeviceToHost);

	//transform
	int bat3=row*col;
	for(int i=0;i<bat3;i++){
	  for(int j=0;j<tupe;j++){
		host_result2[i*tupe+j]=host_result[j*bat3+i];
		}
	}
	cudaMemcpy(d_Tf,host_result2,sizeof(cuComplex)*tupe*row*col,cudaMemcpyHostToDevice);
	
	if(host_result != NULL){
	free(host_result);
	host_result = NULL;
	}

	if(host_result2 != NULL){
	free(host_result2);
	host_result2 = NULL;
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
	//ifft
	for(int i=0;i<bat3;i++){
	if(cufftExecC2C(iplan,d_Tf+i*tupe,d_Tf+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
	

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	if(cufftDestroy(iplan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}

	cuComplex* host_T =(cuComplex*)malloc(sizeof(cuComplex)*tupe*row*col);
	cudaMemcpy(host_T,d_Tf,sizeof(cuComplex)*tupe*bat3,cudaMemcpyDeviceToHost);
	//transform

	for(int i=0;i<bat3;i++){
	  for(int j=0;j<tupe;j++){
		T[j*row*col+i]=host_T[i*tupe+j].x/tupe;
		}
	}

	free(host_T);
	cudaFree(d_Tf);
}
