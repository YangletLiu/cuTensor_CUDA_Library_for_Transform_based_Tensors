#include "tprod.h"
#include "based.h"
void batchedtprod(float* t1,float* t2,float* T,cublasOperation_t t_t1,cublasOperation_t t_t2,int row, int col, int rank, int tupe) {
    int ht = tupe/2 + 1;//half tupe
    int bat1 = row*rank;
    int bat2 = col*rank;
    int bat = bat1 + bat2;
    float *d_t;
    //set stream
    cudaStream_t stream[2];
    #pragma unroll
	for(int i=0;i<2;i++){
		cudaStreamCreate(&stream[i]);
	}
    
    cudaMalloc((void**)&d_t, tupe*bat*sizeof(float));
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,ht*bat*sizeof(cufftComplex));
    cudaMemcpyAsync(d_t,t1,tupe*bat1*sizeof(float),cudaMemcpyHostToDevice,stream[0]);
    cudaMemcpyAsync(d_t+tupe*bat1, t2,tupe*bat2*sizeof(float),cudaMemcpyHostToDevice,stream[1]);
    //tfft
   
    cufftHandle plan =0;
    cufftHandle plan2 =0;


    int n[1] = {tupe};
    int stride = bat1, dist = 1;
    int in[1] = {tupe};
    int on[1] = {ht};
    cufftSetStream(plan,stream[0]);
    cufftSetStream(plan2,stream[1]);
    
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
	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){

		fprintf(stdout,"[%s]:[%d]cufftDestory failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cufftDestroy(plan2)!=CUFFT_SUCCESS){

		fprintf(stdout,"[%s]:[%d]cufftDestory failed!",__FUNCTION__,__LINE__);
		return;
	}
    cudaFree(d_t);
	
	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
    //gemmbatched

    cufftComplex* d_Tf;
    cudaMalloc((void**)&d_Tf,ht*row*col*sizeof(cufftComplex));
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
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    int strA = Am*An;
    int strB = Bm*Bn;
    int strC = Am*Bn;
    if(t_t1==CUBLAS_OP_N && t_t2==CUBLAS_OP_N){
      lda = Am;
      ldb = Bm;
      ldc = Am;
    }else{
        if(t_t1==CUBLAS_OP_N && t_t2==CUBLAS_OP_C){
      lda = Am;
      ldb = Bn;
      ldc = Am;
        }else{
        if( t_t1==CUBLAS_OP_C && t_t2==CUBLAS_OP_N){
      lda = An;
      ldb = Bm;
      ldc = Am;
        }else{
        if(t_t1==CUBLAS_OP_C && t_t2==CUBLAS_OP_C){
      lda = An;
      ldb = Bn;
      ldc = Am;
        }else{
        printf("cublasOperation_t error\n");
        }
        }
        }
    }
    cublasCreate(&handle);
    cublasCgemmStridedBatched(handle, t_t1, t_t2, Am, Bn, Bm,
            &alpha, d_fftData, lda, strA, d_fftData+Am*An*ht, ldb, strB, &beta,
            d_Tf, ldc, strC, ht);
    cublasDestroy(handle);

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
    
    cudaDeviceSynchronize();
    int threads=0;
    int blocks=0;
    int num=bat*tupe;
    if(tupe*bat<512){
       threads=num;
       blocks=1;
     }else{
	threads=512;
	blocks=(num%512 ==0)?num/512:num/512+1;
	}
    fftResultProcess<<<blocks,threads>>>(d_T,num,tupe);
    cudaDeviceSynchronize();
    cudaMemcpy(T,d_T,tupe*bat*sizeof(float),cudaMemcpyDeviceToHost);

    if(cufftDestroy(iplan)!=CUFFT_SUCCESS){

	    fprintf(stdout,"[%s]:[%d]cufftDestory failed!",__FUNCTION__,__LINE__);
	    return;
    }
    cudaFree(d_fftData);
    cudaFree(d_Tf);
    cudaFree(d_T);
//transform
    
    /*for (int i=0; i<tupe*bat; i++)
        T[i] = T[i]/tupe;*/

}
