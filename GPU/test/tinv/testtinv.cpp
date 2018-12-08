#include"inv.h"
#include<time.h>
#include<string.h>
int main(int argc,char* argv[]){
	int m,n,tupe;
	if(argc == 5){
		m=atoi(argv[1]);
		n=atoi(argv[2]);
		tupe=atoi(argv[3]);
		float* invA = (float*)malloc(sizeof(float)*m*n*tupe);
		float* a = (float*)malloc(sizeof(float)*m*n*tupe);
		for(int i=0;i<m*n*tupe;i++){
		a[i]=(float)rand()/(RAND_MAX/100);
	
	//	a[i] = i;
		}
		clock_t start,end;
		if(strcmp("batched",argv[4]) == 0){
			start=clock();
			batchedtinv(a,m,n,tupe,invA);
			end=clock();
			for(int i=0;i<m*n*tupe;i++){
				printf("%f 	",invA[i]);
			}
			
		}else{
            if(strcmp("based",argv[4]) == 0){
                start=clock();
                basedtinv(a,m,n,tupe,invA);
                end=clock();
			for(int i=0;i<m*n*tupe;i++){
				printf("%f 	",invA[i]);
			}
			
            }else
            {
                if(strcmp("streamed",argv[4]) ==0){
                    start=clock();
                    streamedtinv(a,m,n,tupe,invA);
                    end=clock();
			for(int i=0;i<m*n*tupe;i++){
				printf("%f 	",invA[i]);
			}
			
                }else{
                    printf("input error!\n");
                }
            }
        }
		
		double time = (double)(end-start)/CLOCKS_PER_SEC;
		printf("%d %d %d %lf\n",m,n,tupe,time);
	}else{
		fprintf(stdout,"[%s]:[%d] params is error!",__FUNCTION__,__LINE__);
	}	
	return 0;	
}
