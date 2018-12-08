#include "cuNorm.h"
int main()
{
    cuNorm(50);
    printf("warm up!");
    for (int i=20; i<40; i++){
            printf("%d turn\n", i);
            int l=(i+1)*50;
            int n=l*l*l;
            printf("l is %d", l);
            //scanf("%d", &n);
            //double *v = new double[n];
            cuNorm2(l);
    }
    return 0;
}
