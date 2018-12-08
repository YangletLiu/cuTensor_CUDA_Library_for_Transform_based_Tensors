#include "qr.h"
#include "based.h"
void  streamedtqr(float *A,const int m,const int n,const int tupe, cuComplex* Tau)
{	
	
    int bat =m*n;
    cufftComplex* t_f;

    cudaHostAlloc((void**)&t_f,bat*tupe*sizeof(cufftComplex),cudaHostAllocDefault);
    
    //transform t1
	for(int i=0;i<bat;i++){
	   for(int j=0;j<tupe;j++){
		t_f[i*tupe+j].x=A[j*bat+i];
		t_f[i*tupe+j].y=0;
		}
	}
    //set stream for t
    cudaStream_t* stream = (cudaStream_t*)malloc(PLAN1D_SIZE*sizeof(cudaStream_t));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamCreate(&stream[i]);
    }

	//tfft:C2C
	cufftComplex* d_fftData;
	cudaMalloc((void**)&d_fftData,tupe*bat*sizeof(cufftComplex));	
    //process bat
    int bat_num = bat/PLAN1D_SIZE;
    int bat_s = bat%PLAN1D_SIZE;
	cufftHandle * plan=(cufftHandle*)malloc(sizeof(cufftHandle)*PLAN1D_SIZE);
    memset(plan,0,sizeof(cufftHandle));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftPlan1d(&plan[i],tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
	 	fprintf(stdout,"[%s]:[%d] cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;
	}
        cufftSetStream(plan[i],stream[i]);
    }
    if(bat_num > 0){
    for(int j=0;j<bat_num;j++){

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cudaMemcpyAsync(d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,t_f+i*tupe+j*tupe*PLAN1D_SIZE,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
        return;
    }
     }

    #pragma unroll	
	for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(plan[i],d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
            	}
        	}
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cudaMemcpyAsync(t_f+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
        return;
    }
         }
    }

    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,t_f+i*tupe+bat_num*tupe*PLAN1D_SIZE,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]);
    }
    #pragma unroll	
	for(int i=0;i<bat_s;i++){
	if(cufftExecC2C(plan[i],d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,d_fftData+i*tupe+PLAN1D_SIZE*bat_num*tupe,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(t_f+i*tupe+bat_num*tupe*PLAN1D_SIZE,d_fftData+i*tupe+tupe*bat_num*PLAN1D_SIZE,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
    }
    }else{
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(d_fftData+i*tupe,t_f+i*tupe,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]);
    }
    #pragma unroll	
	for(int i=0;i<bat_s;i++){
	if(cufftExecC2C(plan[i],d_fftData+i*tupe,d_fftData+i*tupe,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	    }
	}
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(t_f+i*tupe,d_fftData+i*tupe,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
    }
    }
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
	//transform
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
    */	
    
    cudaMemcpy(d_fftData,t_f2,sizeof(cufftComplex)*bat*tupe,cudaMemcpyHostToDevice);

    for(int i=0;i<PLAN1D_SIZE;i++){	
	    if(cufftDestroy(plan[i])!=CUFFT_SUCCESS){
		    fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		    return;
	    }
    }
		
	if(t_f != NULL){
	cudaFreeHost(t_f);
	t_f = NULL;
	}
	if(t_f2 !=NULL){
	free(t_f2);
	t_f2 = NULL;	
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

    cudaMalloc (( void **)& d_tau , sizeof ( cuComplex ) * tupe * tau);
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
    cudaMalloc (( void **)& devInfo[i] , sizeof ( int ));
	}
       
    cusolverDnCgeqrf_bufferSize(cusolverH[0], m, n, d_fftData, lda, &lwork);

    d_work=(cuComplex**)malloc(tupe*sizeof(cuComplex*));
    #pragma unroll
    for(int i=0;i<tupe;i++){
    cudaMalloc (( void **)& d_work[i] , sizeof ( cuComplex )* lwork );
	}
    int tupe_num= tupe/PLAN1D_SIZE;
    int tupe_s= tupe%PLAN1D_SIZE;
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
	cudaMemcpy(Tau,d_tau,sizeof(cuComplex)* tupe * tau,cudaMemcpyDeviceToHost);

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
    cusolverDnDestroy(cusolverH[i]);
        }
	//Tifft

	//transform
	cuComplex* h_fftData = (cuComplex*)malloc(sizeof(cuComplex) * bat * tupe);
    cuComplex* h_A;
    cudaHostAlloc((void**)&h_A,tupe*bat*sizeof(cuComplex),cudaHostAllocDefault);

    cudaMemcpy(h_fftData, d_fftData, sizeof(cuComplex) *bat *tupe,cudaMemcpyDeviceToHost);

	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		h_A[i*tupe+j]=h_fftData[j*bat+i];
		}
	} 
	
	if(h_fftData != NULL){
	free(h_fftData);
	h_fftData = NULL;
	}
    
	cufftHandle* iplan=(cufftHandle*)malloc(PLAN1D_SIZE*sizeof(cufftHandle));
    
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftPlan1d(&iplan[i],tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;	
	}
    if(cufftSetStream(iplan[i],stream[i]) !=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufft set stream error!",__FUNCTION__,__LINE__);
		return;	
    }
    }
	
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
    bat_num=bat/PLAN1D_SIZE;
    bat_s=bat%PLAN1D_SIZE;
    if(bat_num > 0){
    #pragma unroll
    for(int j=0;j<bat_num;j++){
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
    cudaMemcpyAsync(d_fftData+i*tupe+j*tupe*PLAN1D_SIZE, h_A+i*tupe+j*tupe*PLAN1D_SIZE, sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(iplan[i],d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	    }
    }
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(h_A+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,tupe*sizeof(cuComplex),cudaMemcpyDeviceToHost,stream[i]);
        }
    }
    #pragma unroll
    for(int i=0;i<bat_s;i++){
    cudaMemcpyAsync(d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE, h_A+i*tupe+bat_num*tupe*PLAN1D_SIZE, sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	if(cufftExecC2C(iplan[i],d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	    }
    }
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(h_A+i*tupe+bat_num*tupe*PLAN1D_SIZE,d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,tupe*sizeof(cuComplex),cudaMemcpyDeviceToHost,stream[i]);
    }
    
    }else{
    #pragma unroll
    for(int i=0;i<bat_s;i++){
    cudaMemcpyAsync(d_fftData+i*tupe, h_A+i*tupe, sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	if(cufftExecC2C(iplan[i],d_fftData+i*tupe,d_fftData+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	    }
    }
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(h_A+i*tupe,d_fftData+i*tupe,tupe*sizeof(cuComplex),cudaMemcpyDeviceToHost,stream[i]);
    }
    }
	
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
 	
	//transform
	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		A[j*bat+i]=h_A[i*tupe+j].x/tupe;
		}
	}

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){	
	    if(cufftDestroy(iplan[i])!=CUFFT_SUCCESS){
		    fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		    return;
	    }
    }
	
	cudaFree(d_fftData);
    cudaFree(d_tau);

	cudaFreeHost(h_A);
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){	
    cudaFree(devInfo[i]);
    }
    free(devInfo);
    #pragma unroll
    for(int i=0;i<tupe;i++){	
    cudaFree(d_work[i]);
    }
    free(d_work);
    cudaDeviceReset();
}
