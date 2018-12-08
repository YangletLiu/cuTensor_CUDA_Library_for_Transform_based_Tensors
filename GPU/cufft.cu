#include "fft.h"
void cufft(float *t,int l,int bat,cufftComplex *tf)
{
	cufftComplex *t_f = new cufftComplex[l*bat];
//transform
    for(int i=0;i<bat;i++)
      for(int j=0;j<l;j++){
        t_f[i*l+j].x=t[j*bat+i];
        t_f[i*l+j].y=0;
      }
    cufftComplex *d_fftData;
    // cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));
    CHECK(cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex)));
    // if (cudaGetLastError() != cudaSuccess){
    //   printf(stderr, "Cuda error: Failed to allocate\n");
    //   return; 
    // }

    CHECK(cudaMemcpy(d_fftData,t_f,l*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice));

    for (int i=0; i<bat; i++)
    {
            cufftHandle plan;
            CHECK_CUFFT(cufftPlan1d(&plan,l,CUFFT_C2C,1));
            CHECK_CUFFT(cufftExecC2C(plan,d_fftData + l*i,d_fftData + l*i,CUFFT_FORWARD));
            cudaDeviceSynchronize();
            cufftDestroy(plan);
    }
    CHECK(cudaMemcpy(t_f,d_fftData,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    // cudaFree(d_fftData);
    cudaFree(d_fftData);
//transform
    for(int i=0;i<bat;i++)
          for(int j=0;j<l;j++){
            tf[j*bat+i]=t_f[i*l+j];
          }
    delete[] t_f;
}

