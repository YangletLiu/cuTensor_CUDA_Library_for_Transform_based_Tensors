#include "qr.h"
void  basedtqr(float *A,const int m,const int n,const int tupe, cuComplex* Tau)
{	
	int bat = m*n;
	cufftComplex* t_f = (cufftComplex*)malloc(bat*tupe*sizeof(cufftComplex));
	//transform
	for(int i=0;i<bat;i++){
	   for(int j=0;j<tupe;j++){
		t_f[i*tupe+j].x=A[j*bat+i];
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
	
	cudaMemcpy(d_fftData,t_f2,sizeof(cufftComplex)*bat*tupe,cudaMemcpyHostToDevice);
	
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
   
	// qr
	cusolverDnHandle_t cusolverH;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	cudaError_t cudaStat2 = cudaSuccess ;
	cudaError_t cudaStat3 = cudaSuccess ;
       
     
	cuComplex *d_tau = NULL;
	cuComplex *d_work=NULL;
        int *devInfo=NULL;
	int lda = m;
        int lwork = 0;
        int info_gpu = 0;
	int strid_A=m*n;
	int tau=imin(m,n);
        cusolver_status = cusolverDnCreate(&cusolverH);
    
        cudaStat2 = cudaMalloc (( void **)& d_tau , sizeof ( cuComplex ) * tupe * tau);
        cudaStat3 = cudaMalloc (( void **)& devInfo , sizeof ( int ));
	if(cudaStat2 != cudaSuccess|| cudaStat3 != cudaSuccess){
	}
       
        cusolver_status = cusolverDnCgeqrf_bufferSize(cusolverH, m, n, d_fftData, lda, &lwork);
	cudaDeviceSynchronize();

        cudaMalloc (( void **)& d_work , sizeof ( cuComplex )* lwork );
	 
	for(int i=0;i<tupe;i++){
        cusolver_status = cusolverDnCgeqrf(cusolverH, m, n, d_fftData+i*strid_A, lda, d_tau+i*tau, d_work, lwork, devInfo);
        cudaDeviceSynchronize();
	if(cusolver_status !=CUSOLVER_STATUS_SUCCESS){
	fprintf(stderr,"[%s]:[%d]ERROR!",__FUNCTION__,__LINE__);
	}

	cudaMemcpy(Tau,d_tau,sizeof(cuComplex)* tupe * tau,cudaMemcpyDeviceToHost);
	
        cudaMemcpy (&info_gpu , devInfo , sizeof ( int ) , cudaMemcpyDeviceToHost);
        printf("after geqrf:info_gpu = %d\n", info_gpu);
	}
	//Tifft

	//transform
	cuComplex* h_fftData = (cuComplex*)malloc(sizeof(cuComplex) * bat * tupe);
	cuComplex* h_fftData1 = (cuComplex*)malloc(sizeof(cuComplex) * bat * tupe);
	cudaMemcpy(h_fftData, d_fftData, sizeof(cuComplex) *bat *tupe,cudaMemcpyDeviceToHost);

	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		h_fftData1[i*tupe+j]=h_fftData[j*bat+i];
		}
	} 
	cudaMemcpy(d_fftData, h_fftData1, sizeof(cuComplex)*tupe*bat,cudaMemcpyHostToDevice);
	
	if(h_fftData != NULL){
	free(h_fftData);
	h_fftData = NULL;
	}

	if(h_fftData1 != NULL){
	free(h_fftData1);
	h_fftData1 = NULL;
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
	for(int i=0;i<bat;i++){
	if(cufftExecC2C(iplan,d_fftData+i*tupe,d_fftData+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
    }
    
	cudaDeviceSynchronize();

	cuComplex* h_A = (cuComplex*)malloc(sizeof(cuComplex)*tupe*bat);
	cudaMemcpy(h_A,d_fftData,tupe*bat*sizeof(cuComplex),cudaMemcpyDeviceToHost);
	
	cudaFree(d_fftData);
 	
	//transform
	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		A[j*bat+i]=h_A[i*tupe+j].x/tupe;
		}
	}
	
	cufftDestroy(iplan);
        cudaFree(d_tau);
	free(h_A);
        cudaFree(devInfo);
        cudaFree(d_work);
        cusolverDnDestroy(cusolverH);
}
