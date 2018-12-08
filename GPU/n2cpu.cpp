#include <stdio.h>
#include <math.h>
int main()
{

    int n=5;
    double v[5];
    for (int i=0; i<n; i++)
        v[i] = 1.1;
    double result = 0.0;
    for (int i=0; i<n; i++)
        result += v[i]*v[i];
    result = sqrt(result);
    printf("result is : %lf\n", result);
    return 0;
}
