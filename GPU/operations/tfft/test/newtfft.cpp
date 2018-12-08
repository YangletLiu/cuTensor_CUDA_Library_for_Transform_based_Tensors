#include "fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
void printTensor(cufftComplex *Af, int m, int n, int k){
    for (int i=0; i<k; i++) {
        for (int j=0; j<m; j++){
            for (int z=0; z<n; z++) {
                printf("%lf + %lf ", Af[i*m*n+j*n+z].x, Af[i*m*n+j*n+z].y);
            }
            printf("\n");
        }
        printf("--------------------\n");
    }
}
void printFTensor(float *A, int m, int n, int k){
    for (int i=0; i<k; i++) {
        for (int j=0; j<m; j++){
            for (int z=0; z<n; z++) {
                printf("%lf ", A[i*m*n+j*n+z]);
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
    for (int i=0; i<Am*Ak; i++)
        for (int j=0; j<An; j++){
            scanf("%f", &A[i*An+j]);
        }
    int hAk = Ak/2+1;
    cufftComplex *Af = new cufftComplex[Am*An*hAk];
    Tfft(A, Ak, Am*An, Af);
    printTensor(Af, Am, An, hAk);
    Tifft(A, Ak, Am*An, Af);
    printFTensor(A, Am, An, Ak);
    return 0;
}
