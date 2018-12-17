#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "tprod.h"
#include "fft.h"
#include "warmup.h"
#define random(x) (float(rand()%x))/x
int main(int argc, char** argv)
{
    if(argc==5){
    int Am, An, Ak;
    Am=atoi(argv[1]);
    An=atoi(argv[2]);
    Ak=atoi(argv[3]);
    float *A = new float[Am*An*Ak];
    float *A1;
    cudaHostAlloc((void**)&A1,sizeof(float)*Am*An*Ak,cudaHostAllocDefault);
      int Bm=An, Bn=Am, Bk=Ak;
//    int Bm=Am, Bn=An, Bk=Ak;
    cublasOperation_t t_A1= CUBLAS_OP_N;
    cublasOperation_t t_B1= CUBLAS_OP_N;
    float *B = new float[Bm*Bn*Bk];
    float *B1;
    cudaHostAlloc((void**)&B1,sizeof(float)*Bm*Bn*Bk,cudaHostAllocDefault);
    for (int i=0; i<Am*An*Ak; i++){
        A[i] = random(100);
        A1[i] = random(100);
//	A[i]=i;
//	A1[i]=i;
	}
    for (int i=0; i<Bm*Bn*Bk; i++){
        B[i] = random(100);
        B1[i] = random(100);
//	B[i]=i;
//	B1[i]=i;
	}
    float *C = new float[Am*Bn*Ak];
    float *C1 = new float[Am*Bn*Ak];
    float *C2 = new float[Am*Am*Ak];
    warmup();
    clock_t start, finish;
    if (strcmp(argv[4],"based")==0){
    start = clock();
    basedtprod(A, B, C, Am, Bn, An, Ak);
    finish = clock();
#if 0
    for(int i=0;i<Am*An*Ak;i++)
	{
		printf("%f  ",C[i]);
	}
#endif
    }else{
         if(strcmp(argv[4],"streamed") == 0){
    start = clock();
    streamedtprod(A, B, C, Am, Bn, An, Ak);
    finish = clock();	
    basedtprod(A, B, C1, Am, Bn, An, Ak);
#if 1
    for(int i=0;i<Am*An*Ak;i++)
	{
		printf("%f  ",C[i]-C1[i]);
	}
    printf("\n================================\n");
	for(int i=0;i<Am*An*Ak;i++)
	{
//		printf("%f  ",C[i]);
	}
#endif
    }
	else{
	if(strcmp(argv[4],"batched") == 0){
    start = clock();
    batchedtprod(A1, B1, C, t_A1, t_B1,Am, Bn, An, Ak);
    finish = clock();
 //   basedtprod(A, B, C1, Am, Bn, An, Ak);
#if 0
    for(int i=0;i<Am*An*Ak;i++)
	{
		printf("%f  ",C1[i]);
	}
	
    printf("\n================================\n");
   for(int i=0;i<Am*An*Ak;i++)
	{
		printf("%f  ",C[i]);
	}
#endif
    }else{
    fprintf(stderr,"[%s]:[%d] input error!\n",__FUNCTION__,__LINE__);
	return 0;
	}
           }
    }
    delete A;
    delete B;
    delete C;
    cudaFreeHost(A1);
    cudaFreeHost(B1);
    cudaDeviceReset();

    double time = (double)(finish-start) / CLOCKS_PER_SEC; 
    printf("%d %d %d %d %d %d %lf\n", Am, An, Ak, Bm, Bn, Bk, time);
    return 0;}else{
	printf("input error!\n");
	}
}
