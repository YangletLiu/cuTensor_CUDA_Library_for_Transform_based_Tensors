#include "cuNorm.h"
void cuNorm(int l)
{
    int n = l*l*l;
    double *v = new double[n];
    for (int i=0; i<n; i++)
        v[i] = random(1000);
    clock_t start, finish;
    cublasHandle_t handle ;
    int incx = 1;
    double result;
    double *d_v ;
    start = clock();
    cublasCreate(&handle);
    cudaMalloc ((void**)&d_v, sizeof(double) * n);

    cudaMemcpy(d_v, v, sizeof(double) * n, cudaMemcpyHostToDevice);
    
    cublasDnrm2(handle, n, d_v, incx, &result);
    cudaFree(d_v);
    finish = clock();
    double time = (double)(finish-start) / CLOCKS_PER_SEC; 
    delete v;
    printf("result is : %lf\n", result);
    printf("time is : %lf\n", time);
}
void cuNorm2(int l)
{
    int n = l*l*l;
    double *v = new double[n];
    for (int i=0; i<n; i++)
        v[i] = random(1000);
    clock_t start, finish;
    cublasHandle_t handle ;
    int incx = 1;
    double result;
    double *d_v ;
    start = clock();
    cudaMalloc ((void**)&d_v, sizeof(double) * n);

    cudaMemcpy(d_v, v, sizeof(double) * n, cudaMemcpyHostToDevice);
    
    for (int j=0; j<l; j++)
    {
            cublasCreate(&handle);
            cublasDnrm2(handle, l*l, d_v+l*l*j, incx, &result);
    }
    cudaFree(d_v);
    finish = clock();
    double time = (double)(finish-start) / CLOCKS_PER_SEC; 
    delete v;
    printf("result is : %lf\n", result);
    printf("time is : %lf\n", time);
}
