#include "fft.h"
void basedTfft(float *t,int l,int bat,cufftComplex *tf) {
    cufftComplex *t_f = new cufftComplex[l*bat];
    //transform
    for(int i=0;i<bat;i++)
      for(int j=0;j<l;j++){
        t_f[i*l+j].x=t[j*bat+i];
        t_f[i*l+j].y=0;
      }
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));

    cudaMemcpy(d_fftData,t_f,l*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice);

    cufftHandle plan =0;
    if(cufftPlan1d(&plan,l,CUFFT_C2C, 1)!=CUFFT_SUCCESS){
	fprintf(stdout,"[%s]:[%d] fft cufftPlan1d error!",__FUNCTION__,__LINE__);
	return;	
	}
     
    for (int i=0; i<bat; i++) {
   
        if(cufftExecC2C(plan,(cufftComplex*)(d_fftData+i*l),(cufftComplex*)(d_fftData+i*l),CUFFT_FORWARD)!=CUFFT_SUCCESS){
	fprintf(stdout,"[%s]:[%d] fft cufftExecC2c error!",__FUNCTION__,__LINE__);
	return;
	}
    }

    cudaDeviceSynchronize();
    cudaMemcpy(t_f,d_fftData,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_fftData);
    //transform
    for(int i=0;i<bat;i++)
          for(int j=0;j<l;j++){
            tf[j*bat+i]=t_f[i*l+j];
          }
    delete[] t_f;
}

void basedTifft(float *t, int l, int bat, cufftComplex *tf){
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
     if(cufftPlan1d(&plan,l,CUFFT_C2C,1)!=CUFFT_SUCCESS){
	fprintf(stdout,"[%s]:[%d] ifft cufftPlan1d error!",__FUNCTION__,__LINE__);
	return;
	}
   
    for (int i=0; i<bat; i++) {
        if(cufftExecC2C(plan,(cufftComplex*)(d_fftData+i*l),(cufftComplex*)(d_fftData+i*l),CUFFT_INVERSE)!=CUFFT_SUCCESS){
	fprintf(stdout,"[%s]:[%d] ifft cufftExecC2c error!",__FUNCTION__,__LINE__);
	return;
	}
        }

    cudaDeviceSynchronize();
    cudaMemcpy(t_f,d_fftData,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_fftData);
    //transform
    for(int i=0;i<bat;i++)
          for(int j=0;j<l;j++){
            t[j*bat+i]=t_f[i*l+j].x/l;
          }
    delete[] t_f;
}
