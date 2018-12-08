#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "tprod.h"
#include "fft.h"
#define random(x) (float(rand()%x))/x
void printFTensor(float *A, int m, int n, int k){
    for (int i=0; i<k; i++) {
        for (int j=0; j<m; j++){
            for (int z=0; z<n; z++) {
                printf("%lf ", A[i*m*n+z*m+j]);
            }
            printf("\n");
        }
        printf("--------------------\n");
    }
}
int main(int argc, char** argv)
{
    int Am, An, Ak;
    scanf("%d", &Am);
    scanf("%d", &An);
    scanf("%d", &Ak);
    float *A = new float[Am*An*Ak];
    for (int i=0; i<An*Ak; i++)
        for (int j=0; j<Am; j++){
            scanf("%f", &A[i*Am+j]);
        }
    int Bm, Bn, Bk;
    scanf("%d", &Bm);
    scanf("%d", &Bn);
    scanf("%d", &Bk);
    float *B = new float[Bm*Bn*Bk];
    for (int i=0; i<Bn*Bk; i++)
        for (int j=0; j<Bm; j++){
            scanf("%f", &B[i*Bm+j]);
        }
    float *C = new float[Am*Bn*Ak];
    tprod(A, B, C, Am, Bn, An, Ak);
    printFTensor(C, Am, Bn, Ak);
    delete A;
    delete B;
    delete C;

//for (int j=0; j<Ak; j++)
    //{
    //for (int i = 0; i<Am*Bn*j; i++)
        //printf("%lf ", C[i]);
    //printf("\n");
//}
    return 0;
}
