#include "inv.h"
void basedtinv(float* t,const int m,const int n,const int tupe,float* invA){
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
*/	
    
    if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
		
	if(t_f != NULL){
	free(t_f);
	t_f = NULL;
	}
/*	if(t_f2 !=NULL){
	free(t_f2);
	t_f2 = NULL;	
	}
*/
    //getrf
    int* Pivot;
    int* info;
    int* info_h = (int *)malloc(tupe*sizeof(int));
    cuComplex** Aarray_d;
    cuComplex** Ainv_d;
    cuComplex** Ainv_h;
    cudaMalloc((void**)&Aarray_d,sizeof(cuComplex*));
    cuComplex** Aarray_h=(cuComplex**)malloc(sizeof(cuComplex*));
    cudaMalloc((void**)&Aarray_h[0],sizeof(cuComplex)*bat);
    
    cudaMalloc((void**)&Pivot,tupe*n*sizeof(int));
    cudaMalloc((void**)&info,tupe*sizeof(int));

    cudaMalloc((void**)&Ainv_d,sizeof(cuComplex*));
    Ainv_h=(cuComplex**)malloc(sizeof(cuComplex*));
    cudaMalloc((void**)&Ainv_h[0],bat*sizeof(cuComplex));
    cudaMemcpy(Ainv_d,Ainv_h,sizeof(cuComplex*),cudaMemcpyHostToDevice);
    
	cuComplex* h_fftData = (cuComplex*)malloc(sizeof(cuComplex) * bat * tupe);
    
    cublasHandle_t handle;
    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCreate error!",__FUNCTION__,__LINE__);
		return;
    }
    for(int i=0;i<tupe;i++){
        if(cudaMemcpy(Aarray_h[0],t_f2+i*bat,sizeof(cufftComplex)*bat,cudaMemcpyHostToDevice) != cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cudeMemcpy failed!",__FUNCTION__,__LINE__);
		    return;
        }
        if(cudaMemcpy(Aarray_d,Aarray_h,sizeof(cuComplex*),cudaMemcpyHostToDevice) != cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cudeMemcpy failed!",__FUNCTION__,__LINE__);
		    return;
        }
        if(cublasCgetrfBatched(handle,n,Aarray_d,n,Pivot+i*n,info+i,1) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCgetrf error!",__FUNCTION__,__LINE__);
		return;
        }

        cudaMemcpy(info_h,info,sizeof(int),cudaMemcpyDeviceToHost);
        printf("[ %d ] ",info_h[0]);
        cudaDeviceSynchronize();
        if(cublasCgetriBatched(handle,n,Aarray_d,n,Pivot+i*n,Ainv_d,n,info+i,1) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCgetri error!",__FUNCTION__,__LINE__);
		return;
        }
        cudaMemcpy(h_fftData+i*bat,Ainv_h[0],bat*sizeof(cuComplex),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

    }
    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasDestroy error!",__FUNCTION__,__LINE__);
		return;
    }
    //transform
	cuComplex* h_fftData1 = (cuComplex*)malloc(sizeof(cuComplex) * bat * tupe);

	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		h_fftData1[i*tupe+j]=h_fftData[j*bat+i];
		}
	} 
	cudaMemcpy(d_fftData, h_fftData1, sizeof(cuComplex)*tupe*bat,cudaMemcpyHostToDevice);
    //delete ptr
   if(Aarray_h !=NULL){ 
        cudaFree(Aarray_h[0]);
        Aarray_h=NULL;
        free(Aarray_h);
        Aarray_h=NULL;
    }
    if(Aarray_d !=NULL){ 
        cudaFree(Aarray_d);
        Aarray_d=NULL;
    }
   if(Ainv_h != NULL){
        cudaFree(Ainv_h[0]);
        Ainv_h=NULL;
        free(Ainv_h);
        Ainv_h=NULL;
    }	
    if(Ainv_d != NULL){
        cudaFree(Ainv_d);
        Ainv_d=NULL;
    }
    if(Pivot !=NULL){
        cudaFree(Pivot);
        Pivot=NULL;
    }
    if(info != NULL){
        cudaFree(info);
        info=NULL;
    }
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
	
	//transform
	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		invA[j*bat+i]=h_A[i*tupe+j].x/tupe;
		}
	}
    if(d_fftData != NULL){
        cudaFree(d_fftData);
        d_fftData =NULL;
    }
    if(h_A != NULL){
        free(h_A);
        h_A=NULL;
    }
	
}
