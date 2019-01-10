#include "stdio.h"
#include "stdlib.h"
#include "norm.h"
#include "based.h"
#include "time.h"
#include "string.h"
int main(int argc,char* argv[]){
    int m,n;
    if(argc == 4){
        m=atoi(argv[1]);
        n=atoi(argv[2]);
        float* t=(float*)malloc(m*n*sizeof(float));
        float* v=(float*)malloc(sizeof(float)*m*n);
	float* a=(float*)malloc(sizeof(float)*n);
 
       for(int i=0;i<m*n;i++){
          //  t[i]=(float)rand()/(RAND_MAX/100);
            t[i]=i;
        }

        clock_t start,end;
        if(strcmp("batched",argv[3]) == 0){
            start=clock();
            batchedtnorm(t,m,n,v,a);
            end=clock();
#if 0
	printf("\nV+++++++++++++++++++++++++++++++++++++\n");
	    for(int i=0;i<n*m;i++){
            printf(" %f \n",v[i]);
       	    }
	printf("\na+++++++++++++++++++++++++++++++++++++\n");
	    for(int i=0;i<n;i++){
            printf(" %f \n",a[i]);
       	    }
#endif
	}else{
	if(strcmp("streamed",argv[3]) == 0){
            start=clock();
	    streamedtnorm(t,m,n,v,a);
            end=clock();
#if 0
	printf("\nV+++++++++++++++++++++++++++++++++++++\n");
	    for(int i=0;i<n*m;i++){
            printf(" %f \n",v[i]);
       	    }
	printf("\na+++++++++++++++++++++++++++++++++++++\n");
	    for(int i=0;i<n;i++){
            printf(" %f \n",a[i]);
       	    }
#endif
	}else{
	if(strcmp("based",argv[3]) == 0){
            start=clock();
            basedtnorm(t,m,n,v,a);
            end=clock();
#if 0
	printf("\nV+++++++++++++++++++++++++++++++++++++\n");
	    for(int i=0;i<n*m;i++){
            printf(" %f \n",v[i]);
       	    }
	printf("\na+++++++++++++++++++++++++++++++++++++\n");
	    for(int i=0;i<n;i++){
            printf(" %f \n",a[i]);
       	    }
#endif
	}else{
	printf("params is error!/n");
	}
	}
        }
        
        double time=(double)(end-start)/CLOCKS_PER_SEC;

        printf("%d %d %lf \n",m,n,time);

    }else{
        fprintf(stdout,"[%s]:[%d] params is error!",__FUNCTION__,__LINE__);
        return -1;
    }
}
