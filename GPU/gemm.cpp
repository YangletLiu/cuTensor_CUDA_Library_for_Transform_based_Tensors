#include "head.h"
#include <stdio.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "warmup.h"
int main()
{
    warmup();
    for (int it=0; it<20; it++)
    {
            printf("%d turn\n", it);
            int a = (it+1)*50;
            int Am = a, An = a, Ak = a;
            cuComplex *A = new cuComplex[Am*An*Ak];
            for (int i=0; i<Am*An*Ak; i++)
            {
                A[i].x = random(1000);
                A[i].y = random(1000);
            }
            int Bm = a, Bn = a, Bk = a;
            cuComplex *B = new cuComplex[Bm*Bn*Bk];
            for (int i=0; i<Bm*Bn*Bk; i++)
            {
                B[i].x = random(1000);
                B[i].y = random(1000);
            }
            cuComplex *C = new cuComplex[Am*Bn*Ak];
            delete A;
            delete B;
            delete C;

            finish = clock();
            double time = (double)(finish-start) / CLOCKS_PER_SEC; 
            printf("time is : %lf\n", time);
        }
    //for (int j=0; j<Ak; j++)
            //{
            //for (int i = 0; i<Am*Bn*j; i++)
                //printf("%lf ", C[i]);
            //printf("\n");
        //}
    return 0;
}
