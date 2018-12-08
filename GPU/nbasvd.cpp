#include "head.h"
#include <fstream>
#include "fft.h"
#include "tprod.h"
#include "one_step.h"
#include "norm.h"
#include "Msvd.h"
#include <string>
#include <ctime>
#include <iostream>
void printT(float *T, int a, int b, int c){
    for (int i=0; i<c; i++)
        {for (int j=0; j<a; j++)
            {for (int k=0; k<b; k++)
                std::cout << T[i*a*b+k*a+j] <<" ";
            std::cout<< '\n';}
        std::cout<< "---------------------------\n";}
}

void printTf(cufftComplex *Tf, int a, int b, int c){
    for (int i=0; i<c; i++)
        {for (int j=0; j<a; j++)
            {for (int k=0; k<b; k++)
                std::cout << Tf[i*a*b+k*a+j].x <<"+"<< Tf[i*a*b+k*a+j].y <<"i ";
            std::cout<< '\n';}
        std::cout<< "---------------------------\n";}
}
int main(int argc, char ** argv){
    int rank = 100;
    int a = 200;
    int b = 200;
    int c = 200;
    float *T = new float[a*b*c];
    for (int i=0; i< a*b*c; i++)
        T[i] = random(1000);
    //ifstream red("../data.txt");

    //for (int i=0; i<a*b*c; i++)
        //read >> T[i];
    cufftComplex *Tf = new cufftComplex[a*b*c];
    for (int i=0; i< a*b*c; i++)
        {
                Tf[i].x = random(1000);
                Tf[i].y = random(1000);
        }
         
    //Tfft(T, c, a*b, Tf);

    cufftComplex *Uf = new cufftComplex[a*a*c];
    cufftComplex *Vf = new cufftComplex[b*b*c];
    float *Sf = new float[b*c];
    cudaStream_t *streams = new cudaStream_t[c];
    double *t_arr = new double[c];
    clock_t begin = clock();
    for (int i=0; i<c; i++)
    {
        clock_t t1 = clock();
        Msvd(Tf+i*b*a, Uf+i*a*a, Vf+i*b*b, Sf+i*b, a, b, i);
        clock_t t2 = clock();
        t_arr[i] = double(t2 -t1) /CLOCKS_PER_SEC;
    }
    clock_t end = clock();
    double mainloopE = double(end -begin) / CLOCKS_PER_SEC;
    printf("Time of mainloop %f sec\n", mainloopE);
    for (int i=0; i<c; i++)
        printf("%.3f ", t_arr[i]);
    printf("\n");
    cufftComplex *SS = new cufftComplex[a*b*c];
    for (int i=0; i<c; i++)
        for (int j=0; j<a; j++)
            for (int k=0; k<b; k++)
            {
                SS[i*a*b+k*a+j].y = 0;
                if (j==k)
                {
                    SS[i*a*b+k*a+j].x = Sf[i*b+k];
                }
                else
                {
                    SS[i*a*b+k*a+j].x = 0;
                }
            }
    float *U = new float[a*a*c];
    float *V = new float[b*b*c];
    float *S = new float[b*a*c];
    Tifft(U, c, a*a, Uf);
    Tifft(V, c, b*b, Vf);
    Tifft(S, c, b*a, SS);
    for (int i=0; i<c; i++)
        for (int j=0; j<a; j++)
            for (int k=rank; k<b; k++)
                S[i*a*b+k*a+j] = 0;
    for (int i=0; i<c; i++)
        for (int j=0; j<a; j++)
            for (int k=rank; k<a; k++)
                U[i*a*a+k*a+j] = 0;
    for (int i=0; i<c; i++)
        for (int j=0; j<b; j++)
            for (int k=rank; k<b; k++)
                V[i*b*b+k*b+j] = 0;
    float *US = new float[a*b*c];
    tprod(U, S, US, a, b, a, c);
    float *USV = new float[a*b*c];
    tprod(US, V, USV, a, b, b, c);
    std::cout << "Compress result \n";
    string filename = "CompressK" + to_string(rank) + ".txt";
    ofstream write(filename);
    for (int i=0;i < a*b*c;i++)
        write << USV[i] << " " ;
    float *zero = new float[a*b*c]{0};
    float norm1 = norm(T, USV, a*b*c);
    float norm2 = norm(T, zero, a*b*c);
    std::cout << "rank" << rank << " error is " << norm1/norm2 << "\n";
	cudaDeviceReset();
    delete[] zero;
    zero = nullptr;
    delete T;
    T = nullptr;
    delete Tf;
    Tf = nullptr;
    delete U;
    U = nullptr;
    delete S;
    S = nullptr;
    delete V;
    V = nullptr;
    delete Uf;
    Uf = nullptr;
    delete Sf;
    Sf = nullptr;
    delete Vf;
    Vf = nullptr;
    delete SS;
    SS = nullptr;
    delete US;
    US = nullptr;
    delete USV;
    USV = nullptr;
}
