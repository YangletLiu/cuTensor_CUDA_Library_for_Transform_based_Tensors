#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "tprod.h"
#define random(x) (float(rand()%x))/x

int main(int argc, char** argv) {
    int Am, An, Ak;
    sscanf(argv[1], "%d", &Am);
    sscanf(argv[2], "%d", &An);
    sscanf(argv[3], "%d", &Ak);
    float *A = new float[Am*An*Ak];
    float *A_test = new float[Am*An*Ak];
    for (int i=0; i<Am*An*Ak; i++)
        A[i] = random(1000);
    
#if 0 
    for (int i=0; i<Am*An*Ak; i++)
        printf("%f ",A[i]);
    
    printf("\n+++++++++++++++++++++++++++++++++++++++++++\n");
#endif
    cuComplex *Af = new cuComplex[Am*An*Ak];
    cuComplex *Af2 = new cuComplex[Am*An*Ak];
    clock_t start, finish;
    start = clock();
    if (0 == strcmp("batched", argv[4])) {
        batchedTfft(A, Ak, Am*An, Af);
#if 0 
    for (int i=0; i<Am*An*Ak; i++)
        printf("[%f,%f]  ",Af[i].x,Af[i].y);
    printf("\n+++++++++++++++++++++++++++++++++++++++++++\n");
#endif
        batchedTifft(A_test, Ak, Am*An, Af);
    
#if 0 
    for (int i=0; i<Am*An*Ak; i++)
        printf("[%f]  ",A[i]-A_test[i]);
   
#endif
    }
    else {
        if (0 == strcmp("streamed", argv[4])){
         streamedTfft(A, Ak, Am*An, Af);
#if 0 
    for (int i=0; i<Am*An*Ak; i++)
        printf("[%f,%f]  ",Af[i].x,Af[i].y);
    printf("\n+++++++++++++++++++++++++++++++++++++++++++\n");
#endif
         basedTfft(A, Ak, Am*An, Af2);
#if 0 
    for (int i=0; i<Am*An*Ak; i++)
        printf("[%f,%f]  ",Af[i].x-Af2[i].x,Af[i].y-Af2[i].y);
    printf("\n+++++++++++++++++++++++++++++++++++++++++++\n");
#endif
         streamedTifft(A_test, Ak, Am*An, Af);
    
#if 0 
    for (int i=0; i<Am*An*Ak; i++)
        printf("[%f]  ",A[i]-A_test[i]);
#endif

    }else{
	if(0 == strcmp("based",argv[4])){
         basedTfft(A, Ak, Am*An, Af);
#if 0 
    for (int i=0; i<Am*An*Ak; i++)
        printf("[%f,%f ]  ",Af[i].x,Af[i].y);
    printf("\n+++++++++++++++++++++++++++++++++++++++++++\n");
#endif
  
         basedTifft(A_test, Ak, Am*An, Af);
    
#if 0 
    for (int i=0; i<Am*An*Ak; i++)
        printf("[%f]  ",A[i]-A_test[i]);
#endif
	
    }
        else{
            printf("wrong order\n");
		}	
	}
    }
    
    finish = clock();
    delete A;
    delete Af;
    
    double time = (double)(finish-start) / CLOCKS_PER_SEC; 
    printf("tfft%s of A[%d*%d*%d] take time: %lf\n",argv[4], Am, An, Ak, time);
    
    return 0;
}
