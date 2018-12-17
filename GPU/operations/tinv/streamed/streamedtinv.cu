#include "based.h"
#include "inv.h"
void streamedtinv(float* t,const int m,const int n,const int tupe,float* invA){
#if 1
    //tfft:R2C
	int ht  = tupe/2+1;
	int bat = m*n;
	float* d_t;
	cufftComplex* d_fftData;
	cudaMalloc((void**)&d_t,sizeof(float)*bat*tupe);
	cudaMalloc((void**)&d_fftData,sizeof(cufftComplex)*bat*ht);
	cudaMemcpy(d_t,t,sizeof(float)*bat*tupe,cudaMemcpyHostToDevice);

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
	
	cudaFree(d_t);
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]cufftDestory faile!",__FUNCTION__,__LINE__);
		return;
	}

    cuComplex* t_f2 = (cuComplex*)malloc(sizeof(cuComplex)*bat*ht);
    cudaMemcpy(t_f2,d_fftData,sizeof(cuComplex)*bat*ht,cudaMemcpyDeviceToHost);
    
    if(d_fftData != NULL){
	cudaFree(d_fftData);
	d_fftData=NULL;	
    }
    //set stream for t
    cudaStream_t* stream = (cudaStream_t*)malloc(PLAN1D_SIZE*sizeof(cudaStream_t));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamCreate(&stream[i]);
    }
    //getrf
    int* Pivot;
    int* info;
    int* info_h = (int *)malloc(ht*sizeof(int));
    cuComplex** Aarray_d;
    cuComplex** Ainv_d;
    cuComplex** Ainv_h;
    cudaMalloc((void**)&Aarray_d,sizeof(cuComplex*));
    cuComplex** Aarray_h=(cuComplex**)malloc(sizeof(cuComplex*));
    cudaMalloc((void**)&Aarray_h[0],sizeof(cuComplex)*bat);
    
    cudaMalloc((void**)&Pivot,ht*n*sizeof(int));
    cudaMalloc((void**)&info,ht*sizeof(int));

    cudaMalloc((void**)&Ainv_d,sizeof(cuComplex*));
    Ainv_h=(cuComplex**)malloc(sizeof(cuComplex*));
    cudaMalloc((void**)&Ainv_h[0],bat*sizeof(cuComplex));
    cudaMemcpy(Ainv_d,Ainv_h,sizeof(cuComplex*),cudaMemcpyHostToDevice);
    
    cuComplex* h_fftData = (cuComplex*)malloc(sizeof(cuComplex) * bat * ht);
    
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
    int tupe_num = ht/PLAN1D_SIZE;
    int tupe_s = ht%PLAN1D_SIZE;
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

    cuComplex* d_ifftData;
    cudaMalloc((void**)&d_ifftData,sizeof(cuComplex)*bat*ht);
    cudaMemcpy(d_ifftData,h_fftData,sizeof(cuComplex)*bat*ht,cudaMemcpyHostToDevice);

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
//ifft
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
//	printf("the work size is:%ld G\n",(double)worksize/(1024*1024*1024));
	
	float* d_inv;
	cudaMalloc((void**)&d_inv,sizeof(float)*tupe*bat);

	if(cufftExecC2R(iplan,(cufftComplex*)d_ifftData,d_inv)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
       	int num=bat*tupe;
	float* invA_temp = (float*)malloc(sizeof(float)*tupe*bat);

	cudaMemcpy(invA_temp,d_inv,sizeof(float)*bat*tupe,cudaMemcpyDeviceToHost);
	for(int i=0;i<num;i++){
		invA[i]=invA_temp[i]/tupe;
	}
	cudaFree(d_ifftData);        
	cudaFree(d_inv);
#endif	
}
