#include"sol_v.h"
/**
* solve tensor US*V=T
* US is a tensor , size: m * min(m,n) * k
* T is a tensor , size: m * n * k
* V is a tensor ,size: min(m,n) * n * k 
*/
void solve_v(float* T,float* US,const int m,const int n,const int k,float* V){
int hk = k/2+1;
int min_val = Min(m,n);
cuComplex *tf, *uf, *vf;
tf = (cuComplex*)malloc( sizeof(cuComplex)* m* n* hk);
uf = (cuComplex*)malloc( sizeof(cuComplex)* m * min_val * hk);
vf = (cuComplex*)malloc( sizeof(cuComplex)* min_val * n * hk);

// T take tfft
int bat = m*n;
batchedTfft(T,k,bat,tf);
cudaDeviceSynchronize();

// U take tfft
bat =  m*min_val;
batchedTfft(US,k,bat,uf);
cudaDeviceSynchronize();

// U*X=T
cuComplex *u, *x, *t;
cudaMalloc((void**)&u, sizeof(cuComplex)* m * min_val);
cudaMalloc((void**)&x, sizeof(cuComplex)* min_val * n);
cudaMalloc((void**)&t, sizeof(cuComplex)* m * n);

#pragma unroll
for(int i=0;i< hk;i++){
cudaMemcpy(u,uf+i*m*min_val,sizeof(cuComplex)* m * min_val ,cudaMemcpyHostToDevice);
cudaMemcpy(t,tf+i*m*n,sizeof(cuComplex)* m * n ,cudaMemcpyHostToDevice);
qrsolve(u,t,m,min_val,n,x);
cudaDeviceSynchronize();
cudaMemcpy(vf+i*min_val*n,x,sizeof(cuComplex)* min_val * n,cudaMemcpyDeviceToHost);
}

cudaDeviceSynchronize();
if(u) cudaFree(u);
if(x) cudaFree(x);
if(t) cudaFree(t);
// vf take Tifft
bat = min_val*n;
batchedTifft(V,k,bat,vf);
cudaDeviceSynchronize();

if(tf) free(tf);
if(uf) free(uf);
if(vf) free(vf);
}
