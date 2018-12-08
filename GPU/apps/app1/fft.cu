#include "Tfft.h"


void Tfft(float *t,int l,int bat,cufftComplex *tf)
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

    cufftHandle plan =0;
    CHECK_CUFFT(cufftPlan1d(&plan,l,CUFFT_C2C,bat));
    CHECK_CUFFT(cufftExecC2C(plan,(cufftComplex*)d_fftData,(cufftComplex*)d_fftData,CUFFT_FORWARD));
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(t_f,d_fftData,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    // cudaFree(d_fftData);
    cudaFree(d_fftData);
//transform
    for(int i=0;i<bat;i++)
          for(int j=0;j<l;j++){
            tf[j*bat+i]=t_f[i*l+j];
          }
    delete[] t_f;
}
void mul_cufft(cufftComplex *a,cufftComplex *b,cufftComplex *c){
  c->x += a->x*b->x - a->y*b->y;
  c->y += a->x*b->y + a->y*b->x;
}
void transform(int a,int b,int c,cufftComplex *t,cufftComplex *tt)
{
  for(int i=0;i<c;i++)
    for(int j=0;j<a;j++)
      for(int k=0;k<b;k++){
        tt[i*a*b+j*b+k].x = t[i*a*b+k*a+j].x;
        tt[i*a*b+j*b+k].y = 0 - t[i*a*b+k*a+j].y;
      }
}
void Tifft(float *t,int l,int bat,cufftComplex *tf)
{
  //to be update;use stream?
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
    cufftPlan1d(&plan,l,CUFFT_C2C,bat);
    cufftExecC2C(plan,(cufftComplex*)d_fftData,(cufftComplex*)d_fftData,CUFFT_INVERSE);
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
void printTensor(int m, int n,int k, const float*A)
{
    for(int bt=0;bt<k;bt++){
      for(int row = 0 ; row < m ; row++){
          for(int col = 0 ; col < n ; col++){
              cout<<A[bt*m*n+row + col*m]<<" ";
          }
          cout<<endl;
      }
      cout<<"____________"<<endl;
    }
}
 void TprintTensor(int m, int n,int k, const cufftComplex *A)
{
    for(int bt=0;bt<k;bt++){
      for(int row = 0 ; row < m ; row++){
          for(int col = 0 ; col < n ; col++){
              cout<<A[bt*m*n+row + col*m].x<<"+"<<A[bt*m*n+row + col*m].y<<" ";
          }
          cout<<endl;
      }
      cout<<"____________"<<endl;
    }
}
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
