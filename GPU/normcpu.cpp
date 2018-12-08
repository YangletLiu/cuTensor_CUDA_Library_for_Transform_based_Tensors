#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define random(x) (float(rand()%x))/x

void norm(int n)
{
    double *v = new double[n];
    for (int i=0; i<n; i++)
        v[i] = random(1000);
    clock_t start, finish;
    double c = 0.0;
    start = clock();
    for (int i=0; i<n; i++)
        c = c + v[i] * v[i];
    c = sqrt(c);
    finish = clock();
    double time = (double)(finish-start) / CLOCKS_PER_SEC; 
    delete v;
    printf("result is : %lf\n", c);
    printf("time is : %lf\n", time);
}

int main()
{
    for (int i=0; i<20; i++){
            printf("%d turn\n", i);
            int l=(i+1)*50;
            int n=l*l*l;
            printf("l is %d", l);
            //scanf("%d", &n);
            //double *v = new double[n];
            norm(n);
    }
    return 0;
}
