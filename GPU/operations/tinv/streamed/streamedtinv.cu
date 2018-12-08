#include "based.h"
#include "inv.h"
void streamedtinv(float* t,const int m,const int n,const int tupe,float* invA){
    int bat =m*n;
    cufftComplex* t_f;

    cudaHostAlloc((void**)&t_f,bat*tupe*sizeof(cufftComplex),cudaHostAllocDefault);
    
    //transform t1
	for(int i=0;i<bat;i++){
	   for(int j=0;j<tupe;j++){
		t_f[i*tupe+j].x=t[j*bat+i];
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
//    printf("\n============================\n");
//    for(int i=0;i<bat*tupe;i++){
//    printf("[%f %f]",t_f2[i].x,t_f2[i].y);
//    }	
//    printf("\n============================\n");
    	
    
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
    
    cublasHandle_t* handle = (cublasHandle_t*)malloc(PLAN1D_SIZE*sizeof(cublasHandle_t));
    memset(handle,0,sizeof(cublasHandle_t));
    for(int i=0;i<PLAN1D_SIZE;i++){
    if(cublasCreate(&handle[i]) != CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCreate error!",__FUNCTION__,__LINE__);
		return;
    }
       if( cublasSetStream(handle[i],stream[i]) != CUBLAS_STATUS_SUCCESS){
            fprintf(stdout,"[%s]:[%d] cubalsSetStream error!",__FUNCTION__,__LINE__);
            return;
       }
    }
    int tupe_num = tupe/PLAN1D_SIZE;
    int tupe_s = tupe%PLAN1D_SIZE;
    if( tupe_num > 0){
    for(int j=0;j < tupe_num; j++){
    for(int i=0;i<PLAN1D_SIZE;i++){
        if(cudaMemcpy(Aarray_h[0],t_f2+i*bat+j*bat*PLAN1D_SIZE,sizeof(cufftComplex)*bat,cudaMemcpyHostToDevice) != cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cudeMemcpy failed!",__FUNCTION__,__LINE__);
		    return;
        }
        if(cudaMemcpy(Aarray_d,Aarray_h,sizeof(cuComplex*),cudaMemcpyHostToDevice) != cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cudeMemcpy failed!",__FUNCTION__,__LINE__);
		    return;
        }
        if(cublasCgetrfBatched(handle[i],n,Aarray_d,n,Pivot+i*n+j*n*PLAN1D_SIZE,info+i+j*PLAN1D_SIZE,1) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCgetrf error!",__FUNCTION__,__LINE__);
		return;
        }

        cudaMemcpy(info_h,info,sizeof(int),cudaMemcpyDeviceToHost);
        printf("[ %d ] ",info_h[0]);
        cudaDeviceSynchronize();
        if(cublasCgetriBatched(handle[i],n,Aarray_d,n,Pivot+i*n+j*n*PLAN1D_SIZE,Ainv_d,n,info+i+j*PLAN1D_SIZE,1) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCgetri error!",__FUNCTION__,__LINE__);
		return;
        }
        cudaMemcpy(h_fftData+i*bat+j*bat*PLAN1D_SIZE,Ainv_h[0],bat*sizeof(cuComplex),cudaMemcpyDeviceToHost);
        }
        }
    for(int i=0;i<tupe_s;i++){
        if(cudaMemcpy(Aarray_h[0],t_f2+i*bat+tupe_num*bat*PLAN1D_SIZE,sizeof(cufftComplex)*bat,cudaMemcpyHostToDevice) != cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cudeMemcpy failed!",__FUNCTION__,__LINE__);
		    return;
        }
        if(cudaMemcpy(Aarray_d,Aarray_h,sizeof(cuComplex*),cudaMemcpyHostToDevice) != cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cudeMemcpy failed!",__FUNCTION__,__LINE__);
		    return;
        }
        if(cublasCgetrfBatched(handle[i],n,Aarray_d,n,Pivot+i*n+tupe_num*n*PLAN1D_SIZE,info+i+tupe_num*PLAN1D_SIZE,1) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCgetrf error!",__FUNCTION__,__LINE__);
		return;
        }

        cudaMemcpy(info_h,info,sizeof(int),cudaMemcpyDeviceToHost);
        printf("[ %d ] ",info_h[0]);
        cudaDeviceSynchronize();
        if(cublasCgetriBatched(handle[i],n,Aarray_d,n,Pivot+i*n+tupe_num*n*PLAN1D_SIZE,Ainv_d,n,info+i+tupe_num*PLAN1D_SIZE,1) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCgetri error!",__FUNCTION__,__LINE__);
		return;
        }
        cudaMemcpy(h_fftData+i*bat+tupe_num*bat*PLAN1D_SIZE,Ainv_h[0],bat*sizeof(cuComplex),cudaMemcpyDeviceToHost);

        }
    }else{
    for(int i=0;i<tupe_s;i++){
        if(cudaMemcpy(Aarray_h[0],t_f2+i*bat,sizeof(cufftComplex)*bat,cudaMemcpyHostToDevice) != cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cudeMemcpy failed!",__FUNCTION__,__LINE__);
		    return;
        }
        if(cudaMemcpy(Aarray_d,Aarray_h,sizeof(cuComplex*),cudaMemcpyHostToDevice) != cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cudeMemcpy failed!",__FUNCTION__,__LINE__);
		    return;
        }
        if(cublasCgetrfBatched(handle[i],n,Aarray_d,n,Pivot+i*n,info+i,1) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCgetrf error!",__FUNCTION__,__LINE__);
		return;
        }

        cudaMemcpy(info_h,info,sizeof(int),cudaMemcpyDeviceToHost);
        printf("[ %d ] ",info_h[0]);
        cudaDeviceSynchronize();
        if(cublasCgetriBatched(handle[i],n,Aarray_d,n,Pivot+i*n,Ainv_d,n,info+i,1) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasCgetri error!",__FUNCTION__,__LINE__);
		return;
        }
        cudaMemcpy(h_fftData+i*bat,Ainv_h[0],bat*sizeof(cuComplex),cudaMemcpyDeviceToHost);

        }
    }
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
    if(cublasDestroy(handle[i]) != CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cublasDestroy error!",__FUNCTION__,__LINE__);
		return;
        }
    }
    //transform
	cuComplex* h_fftData1;
    cudaHostAlloc((void**)&h_fftData1,bat*tupe*sizeof(cuComplex),cudaHostAllocDefault);
    
	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		h_fftData1[i*tupe+j]=h_fftData[j*bat+i];
		}
	} 
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

	cuComplex* h_A;
    cudaHostAlloc((void**)&h_A,bat*tupe*sizeof(cuComplex),cudaHostAllocDefault);
	
    cufftHandle* iplan=(cufftHandle*)malloc(PLAN1D_SIZE*sizeof(cufftHandle));
    memset(iplan,0,sizeof(cufftHandle));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftPlan1d(&iplan[i],tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;	
	}
        cufftSetStream(iplan[i],stream[i]);
    }
    bat_num = bat/PLAN1D_SIZE;
    bat_s = bat%PLAN1D_SIZE;
    if(bat_num > 0){ 
	for(int j=0;j<bat_num;j++){
	for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(d_fftData+i*tupe+j*tupe*PLAN1D_SIZE, h_fftData1+i*tupe+j*tupe*PLAN1D_SIZE, sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(iplan[i],d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	    }
    }
    
    for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(h_A+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,tupe*sizeof(cuComplex),cudaMemcpyDeviceToHost,stream[i]);
	}
    }
	for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE, h_fftData1+i*tupe+bat_num*tupe*PLAN1D_SIZE, sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
    for(int i=0;i<bat_s;i++){
	if(cufftExecC2C(iplan[i],d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	    }
    }
    
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(h_A+i*tupe+bat_num*tupe*PLAN1D_SIZE,d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,tupe*sizeof(cuComplex),cudaMemcpyDeviceToHost,stream[i]);
	}
    }else{
	for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(d_fftData+i*tupe, h_fftData1+i*tupe, sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
    for(int i=0;i<bat_s;i++){
	if(cufftExecC2C(iplan[i],d_fftData+i*tupe,d_fftData+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	    }
    }
    
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
		invA[j*bat+i]=h_A[i*tupe+j].x/tupe;
		}
	}
	for(int i=0;i<PLAN1D_SIZE;i++){
	    if(cufftDestroy(iplan[i])!=CUFFT_SUCCESS){
		    fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		    return;
	    }
	    if(cudaStreamDestroy(stream[i])!= cudaSuccess){
		    fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		    return;
	    }

    }
    if(h_fftData1 != NULL){
	cudaFreeHost(h_fftData1);
	h_fftData1 = NULL;
	}
    if(d_fftData != NULL){
        cudaFree(d_fftData);
        d_fftData =NULL;
    }
    if(h_A != NULL){
        cudaFreeHost(h_A);
        h_A=NULL;
    }
	
}
