#include "tprod.h"

void tprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe)
{
  cufftComplex *t1f = new cufftComplex[row*rank*tupe];
  cufftComplex *t2f = new cufftComplex[rank*col*tupe];
  Tfft(t1,tupe,row*rank,t1f);
  Tfft(t2,tupe,rank*col,t2f);
  cufftComplex *Tf = new cufftComplex[row*col*tupe];
  for(int i=0;i<row*col*tupe;i++){
    Tf[i].x=0;
    Tf[i].y=0;
  }
  for(int i=0; i<tupe;i++){
    for(int j=0;j<row;j++){
      for(int k=0;k<col;k++){
        for(int w=0;w<rank;w++){
          mul_cufft(t1f+i*row*rank+w*row+j,t2f+i*rank*col+k*rank+w,Tf+i*row*col+k*row+j);
        }
      }
    }
  }
  delete[] t1f;
  t1f = nullptr;
  delete[] t2f;
  t2f = nullptr;
  Tifft(T,tupe,row*col,Tf);
  delete[] Tf;
  Tf = nullptr;
}