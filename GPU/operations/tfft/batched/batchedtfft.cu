#include "fft.h"
#include "based.h"


void batchedTfft(float *t,int l,int bat,cufftComplex *tf)
{
    int hl = l/2+1;
    float *d_t;
    cudaMalloc((void**)&d_t, l*bat*sizeof(float));
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,hl*bat*sizeof(cufftComplex));
    cudaMemcpy(d_t,t,l*bat*sizeof(float),cudaMemcpyHostToDevice);

    cufftHandle plan;
    int n[1] = {l};
    int stride = bat, dist = 1;
    int in[1] = {l};
    int on[1] = {hl};
    size_t worksize=0;

    if
        (cufftPlanMany(&plan,1, n, in, stride, dist, on, stride, dist,
                       CUFFT_R2C, bat)!=CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
            return; 
        }
    //estimat of the work size
    cufftGetSizeMany(plan,1,n,in,stride,dist,on,stride,dist,CUFFT_R2C,bat,&worksize);
    printf("the work size is:%lf G\n",(double)worksize/(1024*1024*1024));
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
void batchedTifft(float *t,int l,int bat,cufftComplex *tf)
{
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
            fprintf(stderr, "CUIFFT error: Plan creation failed");
            return; 
        }
    if
        (cufftExecC2R(plan,(cufftComplex*)d_fftData, d_t)
         != CUFFT_SUCCESS) {
            fprintf(stderr, "CUIFFT error: EXEC  failed");
            return; 
        }
    cudaDeviceSynchronize();
       	int num=bat*l;
	int threads,blocks;
        if(num<512){
          threads=num;
          blocks=1;
        }else{
	  threads=512;
	  blocks=((num%512 ==0)?num/512:num/512+1);
	}
         fftResultProcess<<<blocks,threads>>>(d_t,num,l);

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

    cudaMemcpy(t,d_t,l*bat*sizeof(float),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_fftData);
    cudaFree(d_t);
}
