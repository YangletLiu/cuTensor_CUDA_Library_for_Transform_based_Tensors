#include "common.h"
#include "fft.h"
#include "head.h"
#include "tprod.h"
#include "one_step.h"
#include "norm.h"
#include <iostream>
#include <cusparse_v2.h>
using namespace std;
void printTensor(int m, int n,int k, const float*A)
{
    for(int bt=0;bt<k;bt++){
      for(int row = 0 ; row < m ; row++){
          for(int col = 0 ; col < n ; col++){
              cout<<A[bt*m*n+row + col*m]<<" ";
          }
          cout<<endl;
      }
      cout<<"____________"<<endl;
    }
}
int main(){
	//init T
	float T[M*N*K];
  float t1[M*r*K];
  // printTensor(M,N,K,T);
  for(int i=0;i<M*r*K;i++)
      t1[i]=random(1000);
  cout << "t1" << endl;
  printTensor(M,r,K,t1);
  float t2[N*r*K];
  for(int i=0;i<N*r*K;i++)
      t2[i]=random(1000);
  cout << "t2" << endl;
  printTensor(M,r,K,t2);
  tprod(t1,t2,T);
  cout << "T" << endl;
  printTensor(M,N,K,T);
  	// for(int i =0; i < M*N*K; i++){
   //  	   if(i%N < r)
   //  		    T[i] = random(1000);
   //  	   else T[i] = 0;
  	// }
  	// printTensor(M,N,K,T);
  	// fft T
  cufftComplex Tf[M*N*K];
	Tfft(T,K,M*N,Tf);
	float p[4]={0.25,0.5,0.75,1.0};
	float T_o[M*N*K];
	// init omega
	float omega[M*N*K];
  	for(int i=0;i<M*N*K;i++){
    	if(random(1000) < p[3])
    	{
    		omega[i]=1;
      		T_o[i]=T[i];
      	}
      	else
      	{
        	omega[i]=0;
        	T_o[i]=0;
      	}
  	}

  	cufftComplex T_of[M*N*K];
  	Tfft(T_o,K,M*N,T_of);
  	//init Y
  	float Y[r*N*K];
  	for(int i = 0; i < r*N*K; i++)
  		Y[i] = random(1000);
  	cufftComplex Y_f[r*N*K];
 	Tfft(Y,K,r*N,Y_f);

  	//trans omega
 	cufftComplex Y_f_trans[r*N*K];
  	transform(r,N,K,Y_f,Y_f_trans);
  	cufftComplex omega_f[M*N*K];
  	Tfft(omega,K,M*N,omega_f);
  	cufftComplex omega_f_k[M*N*K];
  	for (int i = 0; i < M*N*K; i++){
  		omega_f_k[i].x = omega_f[i].x/K;
  		omega_f_k[i].y = omega_f[i].y/K;
  	}
  	cufftComplex omega_f_k_trans[M*N*K];
  	transform(M,N,K,omega_f_k,omega_f_k_trans);
  	cufftComplex T_omega_f_trans[M*N*K];
  	transform(M,N,K,T_of,T_omega_f_trans);
  	cufftComplex X_f[M*r*K],X_f_trans[M*r*K];

  	//data transform
  	// cufftComplex tensor_V_X[N*K*M];
  	// cufftComplex tensor_V_Y[M*K*N];
  	// cufftComplex omega_X[N*K*M*K*M];
  	// cufftComplex omega_Y[M*K*N*K*N];

  	// int Xnnz = M*K*K*N;
  	// int Ynnz = M*K*K*N;
  	// cufftComplex cscValueX[Xnnz];
  	// cufftComplex cscValueY[Ynnz];
  	// cufftComplex * cscRowIndX,*cscColPtrX, *cscRowIndY,*cscColPtrY;
  	// trans_omega(T_of, omega_f_k,omega_X,tensor_V_X, M, N, K);
  	// trans_omega(T_omega_f_trans, omega_f_k_trans,omega_Y,tensor_V_Y, N, M, K);

  	for (int iter = 0; iter < 15; iter++)
  	{
  		cout << "iter = " << iter <<endl;
  		cout << "-------------------------------------" << endl;
  		one_step(T_omega_f_trans,omega_f_k_trans,Y_f_trans,X_f_trans, N, M, K, r);

  		transform(r, M, K, X_f_trans, X_f);

  		one_step(T_of, omega_f_k, X_f, Y_f, M, N, K, r);

  		transform(r, N, K, Y_f, Y_f_trans);
  	}
  	Tifft(Y,K,r*N,Y_f);
  	float X[M*r*K];
  	Tifft(X,K,M*r,X_f);
  	float Test[M*N*K];
  	tprod(X,Y,Test);
  	cout<<"the final result:"<<endl;
  	// printTensor(M,N,K,Test);
  	float zero[M*N*K] = {0};
  	float norm1 = norm(T, Test, M*N*K);
  	float norm2 = norm(T, zero, M*N*K);
  	cout << norm1/norm2 << endl;
	return 0;
}