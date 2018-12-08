#include "tprod.h"
void tprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe) {
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

    cufftHandle plan;
    int n[1] = {tupe};
    int stride = bat1, dist = 1;
    int in[1] = {tupe};
    int on[1] = {ht};
    
    if
        (cufftPlanMany(&plan,1, n, in, stride, dist, on, stride, dist,
                       CUFFT_R2C, bat1)!=CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
            return; 
        }
    if
        (cufftExecR2C(plan, d_t,(cufftComplex*)d_fftData)
         != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: EXEC  failed");
            return; 
        }
    
    cudaDeviceSynchronize();
    stride = bat2;
    if
        (cufftPlanMany(&plan,1, n, in, stride, dist, on, stride, dist,
                       CUFFT_R2C, bat2)!=CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
            return; 
        }
    if
        (cufftExecR2C(plan, d_t+tupe*bat1,d_fftData+ht*bat1)
         != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: EXEC  failed");
            return; 
        }
    cudaFree(d_t);
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
    int strA = Am*An;
    int strB = Bm*Bn;
    int strC = Am*Bn;
    cublasCreate(&handle);
    cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
            &alpha, d_fftData, Am, strA, d_fftData+Am*Bm*ht, Bm, strB, &beta,
            d_Tf, Am, strC, ht);
    cublasDestroy(handle);

    //Tifft

    in[0] = ht;
    on[0] = tupe;
    float* d_T;
    bat = row*col;
    stride = bat;
    cudaMalloc((void**)&d_T, tupe*bat*sizeof(float));
    if
        (cufftPlanMany(&plan,1, n, in, stride, dist, on, stride, dist,
                       CUFFT_C2R, bat)!=CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
            return; 
        }
    if
        (cufftExecC2R(plan,(cufftComplex*)d_Tf, d_T)
         != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: EXEC  failed");
            return; 
        }
    
    cudaDeviceSynchronize();
    cudaMemcpy(T,d_T,tupe*bat*sizeof(float),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_fftData);
    cudaFree(d_Tf);
    cudaFree(d_T);
//transform
    for (int i=0; i<tupe*bat; i++)
        T[i] = T[i]/tupe;

}
/*void streamedtprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe) {
    cufftComplex *t1f = new cufftComplex[row*rank*tupe];
    cufftComplex *t2f = new cufftComplex[rank*col*tupe];
    Tfft(t1,tupe,row*rank,t1f);
    Tfft(t2,tupe,rank*col,t2f);
    cufftComplex *Tf = new cufftComplex[row*col*tupe];
    gemmStrideStreamed(t1f, t2f, Tf, row, rank, tupe, col);
    delete[] t1f;
    delete[] t2f;
    Tifft(T,tupe,row*col,Tf);
    delete[] Tf;
}*/

