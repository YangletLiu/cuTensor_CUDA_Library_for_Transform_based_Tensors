#include "head.h"
#include <fstream>
#include "fft.h"
#include "tprod.h"
#include "one_step.h"
#include "norm.h"
#include <iostream>
using namespace std;

int main(){
	int a = 200;
	int b = a;
	int c = 20;
	int rank = 5;
	//init
	double iStart,iElaps;
	iStart = cpuSecond();
	float *t1 = new float[a*rank*c];
	for(int i=0; i<a*rank*c; i++)
		t1[i] = random(1000);
	float *t2 = new float[rank*b*c];
	for(int i=0; i<rank*b*c; i++)
		t2[i] = random(1000);
	float *T = new float[a*b*c];
    /*float *T_w = new float[720*1280*50];
    ifstream read("../CP_M.txt");
    for (int i =0; i < 720*1280*50; i++)
    {
        read >> T_w[i];
    }
    for (int i=0; i< c;i++)
    {
        for (int j=0; j< a; j++)
            for (int k=0; k< b;k++)
                T[i*a*b+j*b+k] = T_w[i*720*1280+j*720+k];
    }
    delete[] T_w;
    T_w = nullptr;*/

	// cout << "t1" << endl;
	// printTensor(a, rank, c, t1);
	// cout << "t2" << endl;
	// printTensor(rank, b, c, t2);
	tprod(t1, t2, T, a, b, rank, c);
	delete[] t1;
	t1 = nullptr;
	delete[] t2;
	t2 = nullptr;
//error rate versus sample rate
    double err[10];
for (int it=0; it < 10; it++){
	// cout << "T" << endl;
	// printTensor(a, b, c, T);
	// cufftComplex Tf[a*b*c];

	// Tfft(T, c, a*b, Tf);
	// cout << "Tf" << endl;
	// TprintTensor(a, b, c, Tf);
	float *omega =  new float[a*b*c]{0};
	for(int i = 0; i < a*b*c;i++){
		// omega[i] = 1;
		if(random(1000) < 0.1*it+0.1)
			omega[i] = 1;
	}
	// printTensor(a,b,c,omega);
	float *T_omega = new float[a*b*c];
	for(int i=0; i<a*b*c; i++)
		T_omega[i] = omega[i] * T[i];
	// printTensor(a,b,c,T_omega);
	cufftComplex *T_omega_f = new cufftComplex[a*b*c];
	Tfft(T_omega, c, a*b, T_omega_f);
	delete[] T_omega;
	T_omega = nullptr;
	cufftComplex *omega_f = new cufftComplex[a*b*c];
	Tfft(omega, c, a*b, omega_f);
	delete[] omega;
	omega = nullptr;
	// cout << "T_omega_f" << endl;
	// TprintTensor(a, b, c,T_omega_f);
	// cout << "omega_f" << endl;
	// TprintTensor(a, b, c,omega_f);
	float *Y = new float[rank*b*c];
	for(int i=0; i<rank*b*c;i++)
		// Y[i] = 1/(i+1.1);
		Y[i] = random(1000);
	cufftComplex *Y_f = new cufftComplex[rank*b*c];
	Tfft(Y, c, rank*b,Y_f);
	// cout << "Y_f" << endl;
	// TprintTensor(rank, b, c,Y_f);
	cufftComplex *Y_f_trans = new cufftComplex[b*rank*c];
	transform(rank, b, c,Y_f, Y_f_trans);
	cufftComplex *T_omega_f_trans = new cufftComplex[b*a*c];
	transform(a, b, c, T_omega_f, T_omega_f_trans);
	cufftComplex *omega_f_trans = new cufftComplex[a*b*c];
	transform(a, b, c, omega_f, omega_f_trans);
	// cout << "Y_f_trans" << endl;
	// TprintTensor(b , rank, c, Y_f_trans);
	
	cufftComplex *omega_f_k_trans = new cufftComplex[a*b*c];
	for(int i =0;i<a*b*c;i++){
		omega_f_k_trans[i].x = omega_f_trans[i].x / c;
		omega_f_k_trans[i].y = omega_f_trans[i].y / c;
	}
	cufftComplex *omega_f_k = new cufftComplex[a*b*c];
	for(int i =0;i<a*b*c;i++){
		omega_f_k[i].x = omega_f[i].x / c;
		omega_f_k[i].y = omega_f[i].y / c;

	}
	delete[] omega_f_trans;
	omega_f_trans = nullptr;
	delete[] omega_f;
	omega_f = nullptr;
	// cout << "omega_f_k_trans" << endl;
	// TprintTensor(b , a, c, omega_f_k_trans);
	// cout << "T_omega_f_trans" << endl;
	// TprintTensor(b , a, c, T_omega_f_trans);
	cufftComplex *X_f= new cufftComplex[a*rank*c];
    cufftComplex *X_f_trans = new cufftComplex[a*rank*c];
	iElaps = cpuSecond() - iStart;
  	printf("Time of init %f sec\n", iElaps);
	// one_step(T_omega_f_trans,omega_f_k_trans,Y_f_trans,X_f_trans, b, a, c, rank);
	// cout << "X_f_trans" << endl;
	// TprintTensor(rank, a, c, X_f_trans);
	double mainloopS = cpuSecond();
	for (int iter = 0; iter < 10; iter++)
  	{
  		cout << "iter = " << iter <<endl;
  		cout << "-------------------------------------" << endl;
  		double oneS = cpuSecond();
  		one_step(T_omega_f_trans,omega_f_k_trans,Y_f_trans,X_f_trans, b, a, c, rank);
  		transform(rank, a, c, X_f_trans, X_f);
  		one_step(T_omega_f, omega_f_k, X_f, Y_f, a, b, c, rank);

  		transform(rank, b, c, Y_f, Y_f_trans);
  	}
    double mainloopE = cpuSecond() - mainloopS;
    cout << "the main loop takes " << mainloopE << endl;
  	delete[] T_omega_f_trans;
  	T_omega_f_trans = nullptr;
  	delete[] T_omega_f;
  	T_omega_f = nullptr;
  	delete[] omega_f_k_trans;
  	omega_f_k_trans = nullptr;
  	delete[] omega_f_k;
  	omega_f_k = nullptr;
  	Tifft(Y,c,rank*b,Y_f);
    delete[] Y_f_trans;
    Y_f_trans = nullptr;
    delete[] Y_f;
    Y_f= nullptr;
  	float *X = new float[a*rank*c];
    delete[] X_f_trans;
    X_f_trans = nullptr;
  	Tifft(X,c,a*rank,X_f);
    delete[] X_f;
    X_f= nullptr;
  	float *Test = new float[a*b*c];
  	tprod(X,Y,Test, a, b, rank, c);
  	delete[] Y;
  	Y = nullptr;
  	delete[] X;
  	X = nullptr;
  	cudaDeviceReset();
  	cout<<"the final result:"<<endl;
    /*ofstream write("../CP_M_result.txt");
    for (int i=0;i < a*b*c;i++)
        write << Test[i] << " " ;*/
  	// printTensor(M,N,K,Test);
  	float *zero = new float[a*b*c]{0};
  	float norm1 = norm(T, Test, a*b*c);
  	float norm2 = norm(T, zero, a*b*c);
    err[it] = norm1/norm2;
    delete[] zero;
    zero = nullptr;
  	delete[] Test;
  	Test = nullptr;
}
    for (int i=0; i<10;i++)
        cout << err[i] << endl;
  	delete[] T;
  	T = nullptr;
return 0;
}
