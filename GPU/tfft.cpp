#include "fft.h"
#include <stdlib.h>
#include "head.h"
#include <stdio.h>

int main()
{ 
    for (int it=0; it<20; it++)
    {
            printf("%d turn\n", it);
            int a = (it+1)*50;
            float *t = new float[a*a*a];
            for (int i=0; i<a*a*a; i++)
                t[i] = random(1000);
            cufftComplex *tf = new cufftComplex[a*a*a];
            clock_t start, finish;
            start = clock();
            Tfft(t, a, a*a, tf);
            finish  = clock();
            double time = (double)(finish-start) / CLOCKS_PER_SEC; 
            delete t;
            delete tf;
            printf("time is : %lf\n", time);
    }
    return 0;
}
