#include "qr.h"
#include "based.h"
void  streamedtqr(float *A,const int m,const int n,const int tupe, cuComplex* Tau)
{	
	int ht  = tupe/2+1;
	int bat = m*n;
	float* d_t;
	cufftComplex* d_fftData;
	cudaMalloc((void**)&d_t,sizeof(float)*bat*tupe);
	cudaMalloc((void**)&d_fftData,sizeof(cufftComplex)*bat*ht);
	cudaMemcpy(d_t,A,sizeof(float)*bat*tupe,cudaMemcpyHostToDevice);

	//tff
	cufftHandle plan;
	int n_f[1]   = {tupe};
	int stride = bat,dist = 1;
	int in[1]  = {tupe};
	int on[1]  = {ht};
	size_t worksize=0;
	if (cufftPlanMany(&plan,1,n_f,in,stride,dist,on,stride,dist,
				CUFFT_R2C,bat)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	//estimate of the work size
	if(cufftGetSizeMany(plan,1,n_f,in,stride,dist,on,stride,dist,
			CUFFT_R2C,bat,&worksize)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Estimate work size failed!",__FUNCTION__,__LINE__);
		return;
 	}
//	printf("the work size is:%lf G\n",(double)worksize/(1024*1024*1024));

	if(cufftExecR2C(plan,d_t,(cufftComplex*)d_fftData)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	if(d_t !=NULL){
	cudaFree(d_t);
        d_t=NULL;   
        }
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]cufftDestory faile!",__FUNCTION__,__LINE__);
		return;
	}
	
    //set stream for t
    cudaStream_t* stream = (cudaStream_t*)malloc(PLAN1D_SIZE*sizeof(cudaStream_t));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamCreate(&stream[i]);
    }

	// qr
	cusolverDnHandle_t* cusolverH=(cusolverDnHandle_t*)malloc(PLAN1D_SIZE*sizeof(cusolverDnHandle_t));
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

       
     
	cuComplex *d_tau = NULL;
	cuComplex **d_work=NULL;
    int** devInfo=(int**)malloc(PLAN1D_SIZE*sizeof(int*));
	int lda = m;
    int lwork = 0;
	int strid_A=m*n;
	int tau=imin(m,n);
   
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
    cusolver_status = cusolverDnCreate(&cusolverH[i]);
    cusolverDnSetStream(cusolverH[i],stream[i]);
    }

    cudaMalloc (( void **)& d_tau , sizeof ( cuComplex ) * ht * tau);
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
    cudaMalloc (( void **)& devInfo[i] , sizeof ( int ));
	}
       
    cusolverDnCgeqrf_bufferSize(cusolverH[0], m, n, d_fftData, lda, &lwork);

    d_work=(cuComplex**)malloc(ht*sizeof(cuComplex*));
    #pragma unroll
    for(int i=0;i<ht;i++){
    cudaMalloc (( void **)& d_work[i] , sizeof ( cuComplex )* lwork );
	}
    int tupe_num= ht/PLAN1D_SIZE;
    int tupe_s= ht%PLAN1D_SIZE;
    if(tupe_num > 0){
    #pragma unroll
    for(int j=0;j<tupe_num;j++){
    #pragma unroll
	for(int i=0;i<PLAN1D_SIZE;i++){
       if( cusolverDnCgeqrf(cusolverH[i], m, n, d_fftData+i*strid_A+j*strid_A*PLAN1D_SIZE, lda, d_tau+i*tau+j*tau*PLAN1D_SIZE, d_work[i+j*PLAN1D_SIZE], lwork, devInfo[i]) !=CUSOLVER_STATUS_SUCCESS){
		    fprintf(stdout,"[%s]:[%d] cusolverDnCgeqrf error!",__FUNCTION__,__LINE__);
		    return;
       }
        }
     }
    #pragma unroll
	for(int i=0;i<tupe_s;i++){
        cusolver_status = cusolverDnCgeqrf(cusolverH[i], m, n, d_fftData+i*strid_A+tupe_num*strid_A*PLAN1D_SIZE, lda, d_tau+i*tau+tupe_num*tau*PLAN1D_SIZE, d_work[i+tupe_num*PLAN1D_SIZE], lwork, devInfo[i]);
	if(cusolver_status !=CUSOLVER_STATUS_SUCCESS){
	fprintf(stderr,"[%s]:[%d]ERROR!",__FUNCTION__,__LINE__);
		    return;
	        }
        }
    }else{
    #pragma unroll
	for(int i=0;i<tupe_s;i++){
        cusolver_status = cusolverDnCgeqrf(cusolverH[i], m, n, d_fftData+i*strid_A, lda, d_tau+i*tau, d_work[i], lwork, devInfo[i]);
	if(cusolver_status !=CUSOLVER_STATUS_SUCCESS){
	fprintf(stderr,"[%s]:[%d]ERROR!",__FUNCTION__,__LINE__);
		    return;
	        }
    }
    }

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
    cusolverDnDestroy(cusolverH[i]);
        }
	
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
    cudaMemcpy(Tau,d_tau,sizeof(cuComplex)* ht * tau,cudaMemcpyDeviceToHost);
	//Tifft
	cufftHandle iplan =0;
	in[0] = ht;
	on[0] = tupe;
	
	if (cufftPlanMany(&iplan,1,n_f,in,stride,dist,on,stride,dist,
				CUFFT_C2R,bat)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	//estimate of the work size
	if(cufftGetSizeMany(iplan,1,n_f,in,stride,dist,on,stride,dist,
			CUFFT_C2R,bat,&worksize)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Estimate work size failed!",__FUNCTION__,__LINE__);
		return;
 	}
	//printf("the work size is:%ld G\n",(double)worksize/(1024*1024*1024));
	
	float* d_qr;
	cudaMalloc((void**)&d_qr,sizeof(float)*tupe*bat);
	if(cufftExecC2R(iplan,(cufftComplex*)d_fftData,d_qr)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
    
    float* h_qr = (float*)malloc(tupe*bat*sizeof(float));
    cudaMemcpy(h_qr,d_qr,sizeof(float)*tupe*bat,cudaMemcpyDeviceToHost);
    for(int i=0;i<tupe*bat;i++){
	A[i]=h_qr[i]/tupe;
    }	
    free(h_qr);
    cudaFree(d_fftData);
    cudaFree(d_tau);
 
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){	
    cudaFree(devInfo[i]);
    }
    free(devInfo);
    #pragma unroll
    for(int i=0;i<ht;i++){	
    cudaFree(d_work[i]);
    }
    free(d_work);
    cudaDeviceReset();
}
