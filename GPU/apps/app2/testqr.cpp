#if 0
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include "sol_v.h"
#include "svd.h"
#include "tprod.h"
#include "based.h"
using namespace std;
int main(){
int m=2000;
int n=2000;
int k=10;
int min_val=min(m,n);
float* A = new float[m*n*k];
float* res = new float[m*n*k];
float* U = new float[m*min_val*k];
float* V = new float[min_val*n*k];
for(int i=0;i<m*n*k;i++){
	A[i]=(float)rand()/(RAND_MAX/100);
}
for(int i=0;i<m*n*k;i++){
	U[i]=(float)rand()/(RAND_MAX/100);
}
solve_v(A,U,m,n,k,V);

cout<<"V========================"<<endl;
for(int i=0;i<m*n*k;i++){
cout<<V[i]<<" ";
}
cout<< endl;
batchedtprod(U,V,res,CUBLAS_OP_N,CUBLAS_OP_N,m,n,min_val,k);
cout<<"res========================"<<endl;
for(int i=0;i<m*n*k;i++){
printf("%f ",res[i]-A[i]);
}
}
#endif
