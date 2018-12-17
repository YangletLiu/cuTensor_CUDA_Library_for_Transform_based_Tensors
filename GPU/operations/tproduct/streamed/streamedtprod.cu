#include "tprod.h"
#include "based.h"
void streamedtprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe) {
    int ht = tupe/2 + 1;//half tupe
    int bat1 = row*rank;
    int bat2 = col*rank;
    int bat = bat1 + bat2;
    float *d_t;
    
    cudaMalloc((void**)&d_t, tupe*bat*sizeof(float));
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,ht*bat*sizeof(cufftComplex));
    cudaMemcpy(d_t,t1,tupe*bat1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_t+tupe*bat1, t2,tupe*bat2*sizeof(float),cudaMemcpyHostToDevice);
    //tfft
   
    cufftHandle plan =0;
    cufftHandle plan2 =0;


    int n[1] = {tupe};
    int stride = bat1, dist = 1;
    int in[1] = {tupe};
    int on[1] = {ht};
    
    if(cufftPlanMany(&plan,1, n, in, stride, dist, on, stride, dist,
                       CUFFT_R2C, bat1)==CUFFT_ALLOC_FAILED) {
            fprintf(stdout, "[%s]:[%d]CUFFT error: Plan creation failed",__FUNCTION__,__LINE__);
            return; 
        }
    if(cufftExecR2C(plan, d_t,(cufftComplex*)d_fftData)
         != CUFFT_SUCCESS) {
            fprintf(stdout, "[%s]:[%d]CUFFT error: EXEC  failed",__FUNCTION__,__LINE__);
            return; 
        }
    
    int stride2 = bat2;
    if
        (cufftPlanMany(&plan2,1, n, in, stride2, dist, on, stride2, dist,
                       CUFFT_R2C, bat2)!=CUFFT_SUCCESS) {
            fprintf(stdout, "[%s]:[%d]CUFFT error: Plan creation failed",__FUNCTION__,__LINE__);
            return; 
        }
    if
        (cufftExecR2C(plan2, d_t+tupe*bat1,d_fftData+ht*bat1)
         != CUFFT_SUCCESS) {
            fprintf(stdout, "[%s]:[%d]CUFFT error: EXEC  failed",__FUNCTION__,__LINE__);
            return; 
        }
    //destroy plan1 and plan2
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){

		fprintf(stdout,"[%s]:[%d]cufftDestory failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cufftDestroy(plan2)!=CUFFT_SUCCESS){

		fprintf(stdout,"[%s]:[%d]cufftDestory failed!",__FUNCTION__,__LINE__);
		return;
	}
    cudaFree(d_t);
	
    cudaStream_t* stream = (cudaStream_t*)malloc(PLAN1D_SIZE*sizeof(cudaStream_t));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamCreate(&stream[i]);
    }

	//gemmbatched

	cufftComplex* d_Tf;
 	cudaMalloc((void**)&d_Tf,ht*row*col*sizeof(cufftComplex));
	cublasHandle_t* handle=(cublasHandle_t *)malloc(PLAN1D_SIZE*sizeof(cublasHandle_t));
    memset(handle,0,sizeof(cublasHandle_t));
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
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cublasCreate(&handle[i]);
    cublasSetStream(handle[i],stream[i]);
    }
    int tupe_num=ht/PLAN1D_SIZE;
    int tupe_s=ht%PLAN1D_SIZE;
    if(tupe_num > 0){
    #pragma unroll
    for(int j=0;j<tupe_num;j++){
    #pragma unroll
	for(int i=0; i<PLAN1D_SIZE; i++){
	if(cublasCgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
	        &alpha, d_fftData+i*strA+strA*j*PLAN1D_SIZE, Am,d_fftData+strA*ht+i*strB+strB*j*PLAN1D_SIZE, Bm,  &beta,
	        d_Tf+i*strC+j*strC*PLAN1D_SIZE, Am) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cublasCgemm failed!",__FUNCTION__,__LINE__);
		return;
        	}
	  }
    }
    #pragma unroll
	for(int i=0; i<tupe_s; i++){
	if(cublasCgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
	        &alpha, d_fftData+i*strA+strA*tupe_num*PLAN1D_SIZE, Am,d_fftData+strA*ht+i*strB+strB*tupe_num*PLAN1D_SIZE, Bm,  &beta,
	        d_Tf+i*strC+tupe_num*strC*PLAN1D_SIZE, Am) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cublasCgemm failed!",__FUNCTION__,__LINE__);
		return;
	}
	  }
    }else{
    #pragma unroll
	for(int i=0; i<tupe_s; i++){
	if(cublasCgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
	        &alpha, d_fftData+i*strA, Am,d_fftData+strA*ht+i*strB, Bm,  &beta,
	        d_Tf+i*strC, Am) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cublasCgemm failed!",__FUNCTION__,__LINE__);
		return;
	}
	  }
    }
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cublasDestroy(handle[i]);
        cudaStreamDestroy(stream[i]);
    }


	cudaFree(d_fftData);

	//Tifft
    cufftHandle iplan;
    in[0] = ht;
    on[0] = tupe;
    float* d_T;
    bat = row*col;
    stride = bat;
    cudaMalloc((void**)&d_T, tupe*bat*sizeof(float));
    if
        (cufftPlanMany(&iplan,1, n, in, stride, dist, on, stride, dist,
                       CUFFT_C2R, bat)==CUFFT_INTERNAL_ERROR) {
            fprintf(stdout, "[%s]:[%d]CUIFFT error: Plan creation failed",__FUNCTION__,__LINE__);
            return; 
        }
    if
        (cufftExecC2R(iplan,(cufftComplex*)d_Tf, d_T)
         != CUFFT_SUCCESS) {
            fprintf(stdout, "[%s]:[%d]CUIFFT error: EXEC  failed",__FUNCTION__,__LINE__);
            return; 
        }
    float* host_t = (float*)malloc(tupe*bat*sizeof(float));
    cudaMemcpy(host_t,d_T,sizeof(float)*tupe*bat,cudaMemcpyDeviceToHost);
	//transform
    for(int i=0;i<tupe*bat;i++){
	T[i]=host_t[i]/tupe;	
	}
//	for(int i=0;i<bat;i++){
//	  for(int j=0;j<tupe;j++){
//		T[j*bat+i]=host_t[i*tupe+j].x/tupe;
//		}
//	}

    if(stream != NULL){
    free(stream);
    stream=NULL;
    } 
   cudaFree(d_Tf);
   free(host_t);
}
