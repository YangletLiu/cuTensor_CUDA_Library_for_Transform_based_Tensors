#include "mprod.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
int main(int argc,char* argv[]){
	if(argc == 4){
	int Am = atoi(argv[1]);
	int An = atoi(argv[2]);
	int Bn = atoi(argv[3]);
	cuComplex* A = new cuComplex[Am*An];
	cuComplex* B = new cuComplex[An*Bn];
	cuComplex* C = new cuComplex[Am*Bn];
	
	for(int i=0;i<Am*An;i++){
		A[i].x=i;
		A[i].y=i;
	}
	for(int i=0;i<An*Bn;i++){
		B[i].x=i;
		B[i].y=i;
	}
	clock_t start,end;
	
	start = clock();
	mprod(A,B,C,Am,An,Bn);
	end = clock();
	
	double time = (double)(end-start)/CLOCKS_PER_SEC;
	printf("%d %d %lf\n",Am,An,time);	
	return 1;
	}else{
	printf("inputs params is error!\n");
	return -1;
	}
	}
