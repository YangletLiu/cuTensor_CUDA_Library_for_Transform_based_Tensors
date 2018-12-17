#include "head.h"
#include <fstream>
#include "Tfft.h"
#include "tprod.h"
#include "one_step.h"
#include "norm.h"
#include <iostream>
using namespace std;

int main(int argc, char **argv){
        int a = atoi(argv[1]);
        int b = atoi(argv[2]);
        int c = atoi(argv[3]);
        int rank = 5;
#if 0
        float *t1 = new float[a*rank*c];
        for(int i=0; i<a*rank*c; i++)
        	t1[i] = random(1000);
        float *t2 = new float[rank*b*c];
        for(int i=0; i<rank*b*c; i++)
        	t2[i] = random(1000);
        float *T = new float[a*b*c];
        batchedtprod(t1, t2, T, CUBLAS_OP_N, CUBLAS_OP_N, a, b, rank, c);
        delete[] t1;
        t1 = nullptr;
        delete[] t2;
        t2 = nullptr;
#endif
#if 1
        float *T = new float[a*b*c];
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
                    T[i*a*b+j*b+k] = T_w[i*a*b+k*a+j];
        }
        delete[] T_w;
        T_w = nullptr;
#endif
        float *omega =  new float[a*b*c]{0};
        for(int i = 0; i < a*b*c;i++){
            if(random(1000) < 0.5)
                omega[i] = 1;
        }
        float *T_omega = new float[a*b*c];
        for(int i=0; i<a*b*c; i++)
            T_omega[i] = omega[i] * T[i];
        cufftComplex *T_omega_f = new cufftComplex[a*b*c];
        Tfft(T_omega, c, a*b, T_omega_f);
        delete[] T_omega;
        T_omega = nullptr;
        cufftComplex *omega_f = new cufftComplex[a*b*c];
        Tfft(omega, c, a*b, omega_f);
        delete[] omega;
        omega = nullptr;
        float *Y = new float[rank*b*c];
        for(int i=0; i<rank*b*c;i++)
            // Y[i] = 1/(i+1.1);
            Y[i] = random(1000);
        cufftComplex *Y_f = new cufftComplex[rank*b*c];
        Tfft(Y, c, rank*b,Y_f);
        cufftComplex *Y_f_trans = new cufftComplex[b*rank*c];
        transform(rank, b, c,Y_f, Y_f_trans);
        cufftComplex *T_omega_f_trans = new cufftComplex[b*a*c];
        transform(a, b, c, T_omega_f, T_omega_f_trans);
        cufftComplex *omega_f_trans = new cufftComplex[a*b*c];
        transform(a, b, c, omega_f, omega_f_trans);
        
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
        cufftComplex *X_f= new cufftComplex[a*rank*c];
        cufftComplex *X_f_trans = new cufftComplex[a*rank*c];
        double mainloopS = cpuSecond();
        for (int iter = 0; iter < 15; iter++)
        {
//            cout << "iter = " << iter <<endl;
//            cout << "-------------------------------------" << endl;
//            double oneS = cpuSecond();
            one_step(T_omega_f_trans,omega_f_k_trans,Y_f_trans,X_f_trans, b, a, c, rank);
//            double oneE = cpuSecond() - oneS;
//            printf("Time of one_step %f sec\n", oneE);
//            double transS = cpuSecond();

            transform(rank, a, c, X_f_trans, X_f);
//            double transE = cpuSecond() - transS;
//            printf("Time of transform %f sec\n", transE);
            one_step(T_omega_f, omega_f_k, X_f, Y_f, a, b, c, rank);
            transform(rank, b, c, Y_f, Y_f_trans);
        }
        delete[] T_omega_f_trans;
        T_omega_f_trans = nullptr;
        delete[] T_omega_f;
        T_omega_f = nullptr;
        delete[] omega_f_k_trans;
        omega_f_k_trans = nullptr;
        delete[] omega_f_k;
        omega_f_k = nullptr;
        
	double mainloopE = cpuSecond() - mainloopS;
        printf("%d %d %d %f\n",a,b,c,mainloopE);
 
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
        batchedtprod(X,Y,Test, CUBLAS_OP_N,CUBLAS_OP_N,a, b, rank, c);
        delete[] Y;
        Y = nullptr;
        delete[] X;
        X = nullptr;

        cudaDeviceReset();
//        cout<<"the final result:"<<endl;
        string filename = "CP_640_" + to_string(rank) + ".txt";
        ofstream write(filename);
        for (int i=0;i < a*b*c;i++)
            write << Test[i] << " " ;
        float *zero = new float[a*b*c]{0};
        float norm1 = norm(T, Test, a*b*c);
        float norm2 = norm(T, zero, a*b*c);
        delete[] zero;
        zero = nullptr;
        delete[] T;
        T = nullptr;
        delete[] Test;
        Test = nullptr;

        return 0;
}
