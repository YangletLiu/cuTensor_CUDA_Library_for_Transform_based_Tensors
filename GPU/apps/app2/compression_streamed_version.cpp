#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include "svd.h"
#include "tprod.h"
#include "based.h"
using namespace std;
#if 1
int main(int argc, char* argv[]){
    int m=10;
    int n=10;
    int k=10;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    //the data of sumilation
#if 1    
    float* t = new float[m*n*k];
    for(int i=0;i<m*n*k;i++){
//        t[i]=(float)rand()/(RAND_MAX/256);
        t[i]=2;
    }
#if 0
    cout<<"========================="<<endl;
    for(int i=0;i<m*n*k;i++){
      cout<<t[i]<<" ";
    }
#endif

	//real data

#if 0
	int a = m;
	int b = n;
	int c = k;
        float *t = new float[a*b*c];
        float *T_w = new float[a*b*c];
        ifstream read(argv[4]);
        for (int i =0; i < a*b*c; i++)
        {
            read >> T_w[i];
        }
/*        for (int i =0; i < a*b*c; i++)
        {
            cout << T_w[i];
        }
*/
        for (int i=0; i< c;i++)
        {
            for (int j=0; j< a; j++)
                for (int k=0; k< b;k++)
                    t[i*a*b+j*b+k] = T_w[i*a*b+k*a+j];
        }
        delete[] T_w;
        T_w = nullptr;
#endif
#endif

    // take tsvd 
    int u_len = k*m*((m<n)?m:n);
    int v_len = k*n*((m<n)?m:n);
    int s_len = k*((m<n)?m:n);

    float* u = new float[u_len];
    float* v = new float[v_len];
    float* s = new float[s_len];

    clock_t start,end;
    
    start = clock();
    streamedtsvd(t,m,n,k,u,s,v);
 #if 0
    cout<<endl<<"========================="<<endl;
    for(int i=0;i<u_len;i++){
	cout<<u[i]<<" ";
    }
    cout<<endl<<"========================="<<endl;
    for(int i=0;i<v_len;i++){
	cout<<v[i]<<" ";
    }
    cout<<endl<<"========================="<<endl;
    for(int i=0;i<s_len;i++){
	cout<<s[i]<<" ";
    }
#endif   
    //set ratio 
    int MIN = Min(m,n);
//    int ratio = MIN/2;
    int ratio = 0;
    float* d_s;
    cudaMalloc((void**)&d_s,sizeof(float)*s_len);
    cudaMemcpy(d_s,s,sizeof(float)*s_len,cudaMemcpyHostToDevice);
    tubalCompression(d_s,MIN,k,ratio);
    cudaDeviceSynchronize();
#if 0
    cudaMemcpy(s,d_s,sizeof(float)*s_len,cudaMemcpyDeviceToHost);
    cout<<endl<<"========================="<<endl;
    for(int i=0;i<s_len;i++){
	cout<<s[i]<<" ";
    }
#endif
    int row = (m<n)?m:n;
    float* d_s_diagonal;
    float* s_diagonal = new float[row * row * k];
    memset(s_diagonal,0,sizeof(float));
#if 0
cout<<endl<<"======================="<<endl;
    for(int i=0;i<row*row*k;i++){
	cout<<float(s_diagonal[i])<<" ";
    } 
cout<<endl<<"======================="<<endl;
#endif    
    cudaMalloc((void**)&d_s_diagonal,sizeof(float)*row*row*k);
    cudaMalloc((void**)&d_s,sizeof(float)*s_len);

    cudaMemcpy(d_s,s,sizeof(float)*s_len,cudaMemcpyHostToDevice);
    
    batcheddiagmat(d_s,row,k,d_s_diagonal);
    cudaDeviceSynchronize();
    cudaMemcpy(s_diagonal,d_s_diagonal,sizeof(float)*row*row*k,cudaMemcpyDeviceToHost);
#if 0 
cout<<endl<<"======================="<<endl;
    for(int i=0;i<row*row*k;i++){
	cout<<(float)s_diagonal[i]<<" ";
    } 
cout<<endl<<"======================="<<endl;
#endif
    delete[] s;
    cudaFree(d_s);
    cudaFree(d_s_diagonal);
    
    // u*s*v' 
    float* u_s = new float[m*row*k];
    float* u_s_vt = new float[m*n*k];
    batchedtprod(u,s_diagonal,u_s,CUBLAS_OP_N,CUBLAS_OP_N,m,row,row,k);
//    for(int i=0;i<m*row*k;i++){
//	cout<<u_s[i]<<" ";
//    }
//    cout<<endl;
    batchedtprod(u_s,v,u_s_vt,CUBLAS_OP_N,CUBLAS_OP_C,m,n,row,k);
//    cout<<endl;
    for(int i=0;i<m*n*k;i++){
    cout<<(u_s_vt[i]-t[i])<<' ';
    }
    cout<<endl;
    end = clock();
    cout << float(end-start) / CLOCKS_PER_SEC;
    delete[] u_s;
    delete[] u_s_vt;
    return 1;
}
#endif
