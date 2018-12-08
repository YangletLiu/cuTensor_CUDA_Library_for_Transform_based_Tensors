#include "fft.h"


void Tfft(float *t,int l,int bat,cufftComplex *tf)
{
    int hl = l/2+1;
    float *d_t;
    cudaMalloc((void**)&d_t, l*bat*sizeof(float));
    cufftComplex *d_fftData;
    // cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));
    cudaMalloc((void**)&d_fftData,hl*bat*sizeof(cufftComplex));
    // if (cudaGetLastError() != cudaSuccess){
    //   printf(stderr, "Cuda error: Failed to allocate\n");
    //   return; 
    // }

    cudaMemcpy(d_t,t,l*bat*sizeof(float),cudaMemcpyHostToDevice);

    cufftHandle plan;
    int n[1] = {l};
    int stride = bat, dist = 1;
    int in[1] = {l};
    int on[1] = {hl};
    
    if
        (cufftPlanMany(&plan,1, n, in, stride, dist, on, stride, dist,
                       CUFFT_R2C, bat)!=CUFFT_SUCCESS) {
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
    cudaMemcpy(tf,d_fftData,hl*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_t);
    cudaFree(d_fftData);
}
void Tifft(float *t,int l,int bat,cufftComplex *tf)
{
  //to be update;use stream?
//transform
    int hl = l/2+1;
    float *d_t;
    cudaMalloc((void**)&d_t, l*bat*sizeof(float));
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,hl*bat*sizeof(cufftComplex));
    cudaMemcpy(d_fftData,tf,hl*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice);

    cufftHandle plan =0;
    int n[1] = {l};
    int stride = bat, dist = 1;
    int in[1] = {hl};
    int on[1] = {l};
    
    if
        (cufftPlanMany(&plan,1, n, in, stride, dist, on, stride, dist,
                       CUFFT_C2R, bat)!=CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
            return; 
        }
    if
        (cufftExecC2R(plan,(cufftComplex*)d_fftData, d_t)
         != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: EXEC  failed");
            return; 
        }
    cudaDeviceSynchronize();
    cudaMemcpy(t,d_t,l*bat*sizeof(float),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_fftData);
    cudaFree(d_t);
//transform
    for (int i=0; i<l*bat; i++)
        t[i] = t[i]/l;
}
void streamedTfft(float *t,int l,int bat,cufftComplex *tf) {
	cufftComplex *t_f = new cufftComplex[l*bat];
//transform
    for(int i=0;i<bat;i++)
      for(int j=0;j<l;j++){
        t_f[i*l+j].x=t[j*bat+i];
        t_f[i*l+j].y=0;
      }
    cufftComplex *d_fftData;
    // cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));
    cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));
    // if (cudaGetLastError() != cudaSuccess){
    //   printf(stderr, "Cuda error: Failed to allocate\n");
    //   return; 
    // }

    cudaMemcpy(d_fftData,t_f,l*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice);

    cufftHandle plan =0;
    cufftPlan1d(&plan,l,CUFFT_C2C, 1);
    cudaStream_t *streams = (cudaStream_t *) malloc(bat*sizeof(cudaStream_t));
    for (int i=0; i<bat; i++)
        cudaStreamCreate(&streams[i]);
    for (int i=0; i<bat; i++) {
        cufftSetStream(plan, streams[i]);
        cufftExecC2C(plan,(cufftComplex*)(d_fftData+i*l),(cufftComplex*)(d_fftData+i*l),CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(t_f,d_fftData,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    free(streams);
    cudaFree(d_fftData);
//transform
    for(int i=0;i<bat;i++)
          for(int j=0;j<l;j++){
            tf[j*bat+i]=t_f[i*l+j];
          }
    delete[] t_f;
}

void streamedTifft(float *t, int l, int bat, cufftComplex *tf){
  cufftComplex *t_f = new cufftComplex[l*bat];
//transform
    for(int i=0;i<bat;i++)
      for(int j=0;j<l;j++){
        t_f[i*l+j]=tf[j*bat+i];
      }
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));
    cudaMemcpy(d_fftData,t_f,l*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice);

    cufftHandle plan =0;
    cufftPlan1d(&plan,l,CUFFT_C2C,1);
    cudaStream_t *streams = (cudaStream_t *) malloc(bat*sizeof(cudaStream_t));
    for (int i=0; i<bat; i++)
        cudaStreamCreate(&streams[i]);
    for (int i=0; i<bat; i++) {
        cufftSetStream(plan, streams[i]);
        cufftExecC2C(plan,(cufftComplex*)(d_fftData+i*l),(cufftComplex*)(d_fftData+i*l),CUFFT_INVERSE);
        }
    cudaDeviceSynchronize();
    cudaMemcpy(t_f,d_fftData,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    free(streams);
    cudaFree(d_fftData);
//transform
    for(int i=0;i<bat;i++)
          for(int j=0;j<l;j++){
            t[j*bat+i]=t_f[i*l+j].x/l;
          }
    delete[] t_f;
}
