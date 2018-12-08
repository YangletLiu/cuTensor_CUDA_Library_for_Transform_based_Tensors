#include "stdio.h"
#include "stdlib.h"
#include "norm.h"
#include "time.h"
#include "string.h"
int main(int argc,char* argv[]){
    int m,n,tube;
    if(argc == 5){
        m=atoi(argv[1]);
        n=atoi(argv[2]);
        tube=atoi(argv[3]);
        float* t=(float*)malloc(m*n*tube*sizeof(float));
        float* result=(float*)malloc(sizeof(float));
        for(int i=0;i<m*n*tube;i++){
          //  t[i]=(float)rand()/(RAND_MAX/100);
            t[i]=1;
        }
        clock_t start,end;
        if(strcmp("batched",argv[4]) == 0){
            start=clock();
            batchedtnorm(t,m,n,tube,result);
            end=clock();
            printf(" norm: %f \n",*result);
        }else{
        }
        
        double time=(double)(end-start)/CLOCKS_PER_SEC;
        printf("%d %d %d %lf \n",m,n,tube,time);

    }else{
        fprintf(stdout,"[%s]:[%d] params is error!",__FUNCTION__,__LINE__);
        return -1;
    }
}
