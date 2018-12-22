#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include "sol_v.h"
#include "svd.h"
#include "tprod.h"
#include "based.h"
using namespace std;

int main(int argc, char* argv[]){
    int m=2;
    int n=2;
    int k=2;
    //the data of sumilation
#if 1    
    float* t = new float[m*n*k];
    for(int i=0;i<m*n*k;i++){
       // t[i]=(float)rand()/(RAND_MAX/256);
        t[i]=2;
    }
    cout<<"========================="<<endl;
    for(int i=0;i<m*n*k;i++){
      cout<<t[i]<<" ";
    }

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

    batchedtsvd(t,m,n,k,u,s);

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
    //sort s
    float* tag = new float[s_len];
    for(int i=0;i<s_len;i++){
        tag[i]=i;
    }

    int flag = 0;
    int length = s_len;
    while( flag > 0 ){
        flag = 0;
        for(int i=1;i<length;i++){
            if(s[i] < s[i-1]){
                int temp=s[i];
                s[i] = s[i-1];
                s[i-1] = temp;

                int temp_tag = tag[i];
                tag[i] = tag[i-1];
                tag[i-1] = temp_tag;
                
                flag = i;
            }
        }
        length = flag;
    }
    cout<<endl<<"========================="<<endl;
    for(int i=0;i<s_len;i++){
	cout<<s[i]<<" ";
    }

    //ratio
//    int r= s_len-s_len/2;
    int r=0;
    for(int i=0;i<r;i++){
        s[i]=0;
    } 

    flag = 0;
    length = s_len;
    while( flag > 0 ){
        flag = 0;
        for(int i=1;i<length;i++){
            if(tag[i] < tag[i-1]){
                int temp=s[i];
                s[i] = s[i-1];
                s[i-1] = temp;

                int temp_tag = tag[i];
                tag[i] = tag[i-1];
                tag[i-1] = temp_tag;
                
                flag = i;
            }
        }
        length = flag;
    }
    cout<<endl<<"========================="<<endl;
    for(int i=0;i<s_len;i++){
	cout<<s[i]<<" ";
    }

    delete[] tag;
    
    int row = (m<n)?m:n;
    float* d_s;
    float* d_s_diagonal;
    float* s_diagonal = new float[row * row * k];
    memset(s_diagonal,0,sizeof(float));
cout<<endl<<"======================="<<endl;
    for(int i=0;i<row*row*k;i++){
	cout<<float(s_diagonal[i])<<" ";
    } 
cout<<endl<<"======================="<<endl;
    
    cudaMalloc((void**)&d_s_diagonal,sizeof(float)*row*row*k);
    cudaMalloc((void**)&d_s,sizeof(float)*s_len);

    cudaMemcpy(d_s,s,sizeof(float)*s_len,cudaMemcpyHostToDevice);
    
    batcheddiagmat(d_s,row,k,d_s_diagonal);

    cudaMemcpy(s_diagonal,d_s_diagonal,sizeof(float)*row*row*k,cudaMemcpyDeviceToHost);
cout<<endl<<"======================="<<endl;
    for(int i=0;i<row*row*k;i++){
	cout<<(float)s_diagonal[i]<<" ";
    } 
cout<<endl<<"======================="<<endl;
    delete[] s;
    cudaFree(d_s);
    cudaFree(d_s_diagonal);
    
    // u*s*v' 
    float* u_s = new float[m*row*k];
    float* u_s_vt = new float[m*n*k];
    batchedtprod(u,s_diagonal,u_s,CUBLAS_OP_N,CUBLAS_OP_N,m,row,row,k);
    for(int i=0;i<m*row*k;i++){
	cout<<u_s[i]<<" ";
    }
    cout<<endl;
cout<<endl<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl;
    //solve v
    solve_v(t,u_s,m,n,k,v);
cout<<endl<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl;

//    batchedtprod(u_s,v,u_s_vt,CUBLAS_OP_N,CUBLAS_OP_C,m,n,row,k);
    batchedtprod(u_s,v,u_s_vt,CUBLAS_OP_N,CUBLAS_OP_N,m,n,row,k);
    cout<<endl;
    for(int i=0;i<m*n*k;i++){
    cout<<u_s_vt[i]<<' ';
    }
    cout<<endl;
    end = clock();
    cout << float(end-start) / CLOCKS_PER_SEC;
    delete[] u_s;
    delete[] u_s_vt;
    return 1;
}
