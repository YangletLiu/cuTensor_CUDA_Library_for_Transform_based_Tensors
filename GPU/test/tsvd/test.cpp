#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "svd.h"
#include "based.h"
int  main(int argc,char* argv[]){
    if(argc==5){
        int m = atoi(argv[1]);
        int n = atoi(argv[2]);
        int tupe = atoi(argv[3]);
        float* t = new float[m*n*tupe];
        for(int i=0;i<m*n*tupe;i++){
           t[i]=rand()/(RAND_MAX/10);
//            t[i]=i;
        }

       // cufftComplex* fftout=(cufftComplex*)malloc(sizeof(cufftComplex)*m*n*(tupe/2+1));
        int u_len = tupe*m*((m<n)?m:n);
        int v_len = tupe*n*((m<n)?m:n);
        int s_len = tupe*((m<n)?m:n);

        float* u = new float[u_len]();
        float* v = new float[v_len]();
        float* s = new float[s_len]();

        float* test_u = new float[u_len]();
        float* test_v = new float[v_len]();
        float* test_s = new float[s_len]();
        clock_t start,end;

//	int ht=tupe/2+1;
//	int hs_len=((2*m<2*n)?2*m:2*n);
//	float* h_s = new float[ht*hs_len];
    
        if(strcmp("streamed",argv[4]) == 0){ 
            start = clock();    
            streamedtsvd(t,m,n,tupe,u,s,v);
            end = clock();
            basedtsvd(t,m,n,tupe,test_u,test_s,test_v);
#if 0
printf("\n++++++++++++++++++++\n");
         for(int i=0;i<u_len;i++){
               printf("%f\n",u[i]-test_u[i]);
               }
            printf("\n++++++++++++++++++++\n");
           for(int i=0;i<v_len;i++){
               printf("%f\n",v[i]-test_v[i]);
               }
            printf("\n++++++++++++++++++++\n");
          for(int i=0;i<s_len;i++){
               printf("%f\n",s[i]-test_s[i]);
               }
#endif
        }else{
            if(strcmp("batched",argv[4]) == 0){ 
            //  batchedsvd(t,m,n,tupe,u,s,v);
	float* host_u, *host_s;
	cudaHostAlloc((void**)&host_u,sizeof(float)*m*n*tupe,cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_s,sizeof(float)*s_len,cudaHostAllocDefault);
            start = clock();    
                batchedtsvd(t,m,n,tupe,host_u,host_s);
            end = clock();
//            printf("\n++++++++++++++++++++\n");
//	printf("***************U\n");
//        for(int i=0;i<m*n*tupe;i++){
//            printf(" %f 	",host_u[i]);
//        }
//            printf("\n++++++++++++++++++++\n");
//	printf("***************S\n");
//            for(int i=0;i<s_len;i++){
//                printf("%f\n",host_s[i]);
//                    }
	cudaFreeHost(host_u);
	cudaFreeHost(host_s);
    
        }else{

            if(strcmp("based",argv[4]) == 0){ 
            start = clock();    
               basedtsvd(t,m,n,tupe,u,s,v); 
            end = clock();
#if 0      
         printf("\n++++++++++++++++++++\n");
          for(int i=0;i<u_len;i++){
                printf("%f\n",u[i]);
                }
            printf("\n++++++++++++++++++++\n");
            for(int i=0;i<v_len;i++){
                printf("%f\n",v[i]);
                }
/*           printf("\n++++++++++++++++++++\n");
            for(int i=0;i<s_len;i++){
                printf("%f\n",s[i]);
                    }*/
#endif          
          }else{
                    printf("argv[4] should be based or batched or stramed\n");
                    exit(-1);
                }
            }

        }

        delete[] u;
        delete[] v;
        delete[] s;
        double time = double(end-start) / CLOCKS_PER_SEC;
        printf("%d %d %d %lf \n",m,n,tupe,time);
    }else{
    fprintf(stdout,"[%s]:[%d] params unput error! ",__FUNCTION__,__LINE__);
        exit(-1);
    }
    return 1;
    }
