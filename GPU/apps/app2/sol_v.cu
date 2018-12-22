#include"sol_v.h"
/**
* solve tensor US*V=T
* US is a tensor , size: m * min(m,n) * k
* T is a tensor , size: m * n * k
* V is a tensor ,size: min(m,n) * n * k 
*/
void solve_v(float* T,float* US,const int m,const int n,const int k,float* V){
int hk = k/2+1;
int min = Min(m,n);
cuComplex *tf, *uf, *vf;
tf = (cuComplex*)malloc( sizeof(cuComplex)* m* n* hk);
uf = (cuComplex*)malloc( sizeof(cuComplex)* m * min * hk);
vf = (cuComplex*)malloc( sizeof(cuComplex)* min * n * hk);

// T take tfft
int bat = m*n;
batchedTfft(T,k,bat,tf);

// U take tfft
bat =  m*min;
batchedTfft(US,k,bat,uf);

// U*X=T
cuComplex *u, *x, *t;
cudaMalloc((void**)&u, sizeof(cuComplex)* m * min);
cudaMalloc((void**)&x, sizeof(cuComplex)* min * n);
cudaMalloc((void**)&t, sizeof(cuComplex)* m * n);

#pragma unroll
for(int i=0;i< hk;i++){
cudaMemcpy(u,uf+i*m*min,sizeof(cuComplex)* m * min ,cudaMemcpyHostToDevice);
cudaMemcpy(t,tf+i*m*n,sizeof(cuComplex)* m * n ,cudaMemcpyHostToDevice);
qrsolve(u,t,m,min,n,x);
cudaMemcpy(vf+i*min*n,x,sizeof(cuComplex)* min * n,cudaMemcpyDeviceToHost);
}
if(u) cudaFree(u);
if(x) cudaFree(x);
if(t) cudaFree(t);
// vf take Tifft
bat = min*n;
batchedTifft(V,k,bat,vf);

if(tf) free(tf);
if(uf) free(uf);
if(vf) free(vf);
}
