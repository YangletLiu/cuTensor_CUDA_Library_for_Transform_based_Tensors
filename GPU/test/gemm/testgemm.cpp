#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "tprod.h"
#define random(x) (float(rand()%x))/x
int main(int argc, char** argv)
{
    if(argc==8){
    int Am, An, Ak;
    sscanf(argv[1], "%d", &Am);
    sscanf(argv[2], "%d", &An);
    sscanf(argv[3], "%d", &Ak);
    cuComplex *A = new cuComplex[Am*An*Ak];
    int Bm, Bn, Bk;
    sscanf(argv[4], "%d", &Bm);
    sscanf(argv[5], "%d", &Bn);
    sscanf(argv[6], "%d", &Bk);
    cuComplex *B = new cuComplex[Bm*Bn*Bk];
    for (int i=0; i<Am*An*Ak; i++)
    {
        A[i].x = random(1000);
        A[i].y = random(1000);
    }
    for (int i=0; i<Bm*Bn*Bk; i++)
    {
        B[i].x = random(1000);
        B[i].y = random(1000);
    }
    cuComplex *C = new cuComplex[Am*Bn*Ak];
    clock_t start, finish;
    start = clock();
    if (0 == strcmp("batched", argv[7]))
        gemmStrideBatched(A, B, C, Am, An, Ak, Bn);
    else {
        if (0 == strcmp("streamed", argv[7]))
            gemmStrideStreamed(A, B, C, Am, An, Ak, Bn);
        else{
	if (0 == strcmp("based",argv[7])){
            gemmStrideBased(A, B, C, Am, An, Ak, Bn);
	}else{
	    printf("argv[7] is :based or streamed or batched\n");
	}
	}
    }
    
    finish = clock();
    delete A;
    delete B;
    delete C;

    double time = (double)(finish-start) / CLOCKS_PER_SEC; 
    printf("gemm%s of A[%d*%d*%d] and B[%d*%d*%d] take time: %lf\n",argv[7], Am, An, Ak, Bm, Bn, Bk, time);
//for (int j=0; j<Ak; j++)
    //{
    //for (int i = 0; i<Am*Bn*j; i++)
        //printf("%lf ", C[i]);
    //printf("\n");
//}
    return 0;}else{
	printf("argv[1]~argv[7] must be input data of int!\n");
	}
}
