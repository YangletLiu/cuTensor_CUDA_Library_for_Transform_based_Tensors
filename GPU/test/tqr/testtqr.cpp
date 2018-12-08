#include"batchedtqr.h"
#include"qr.h"
#include<time.h>
#include<string.h>
int main(int argc,char* argv[]){
	int m,n,tupe;
	if(argc == 5){
		m=atoi(argv[1]);
		n=atoi(argv[2]);
		tupe=atoi(argv[3]);
		cuComplex* tau=(cuComplex*)malloc(sizeof(cuComplex)*m*n*tupe);
		cuComplex* test_tau=(cuComplex*)malloc(sizeof(cuComplex)*m*n*tupe);
		float* a = (float*)malloc(sizeof(float)*m*n*tupe);
		float* test_a = (float*)malloc(sizeof(float)*m*n*tupe);
		for(int i=0;i<m*n*tupe;i++){
			a[i]=(float)rand()/(RAND_MAX/100);
		}
		for(int i=0;i<m*n*tupe;i++){
		 test_a[i]=a[i];
		}
		clock_t start,end;
		if(strcmp("batched",argv[4]) == 0){
			start=clock();
			batchedtqr(a,m,n,tupe,tau);
			end=clock();
#if 0		
        for(int i=0;i<m*n*tupe;i++){
				printf("%f 	",a[i]);
			}
		printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++\n");
			for(int i=0;i<min(m,n)*tupe;i++){
				printf("%f %f  ",tau[i].x,tau[i].y);
	    }
#endif
        }else{
	         	if(strcmp("based",argv[4]) == 0){
			start=clock();
		 	basedtqr(a,m,n,tupe,tau);
			end=clock();
#if 0
        for(int i=0;i<m*n*tupe;i++){
				printf("%f 	",a[i]);
			}
		printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++\n");
			for(int i=0;i<min(m,n)*tupe;i++){
				printf("%f %f  ",tau[i].x,tau[i].y);
			}
#endif
            }else{
				
			 if(strcmp("streamed",argv[4]) == 0){
			start=clock();
		 	streamedtqr(a,m,n,tupe,tau);
			end=clock();
#if 0
		 	basedtqr(test_a,m,n,tupe,test_tau);
        for(int i=0;i<m*n*tupe;i++){
				printf("%f 	",a[i]-test_a[i]);
			}
		printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++\n");
			for(int i=0;i<min(m,n)*tupe;i++){
				printf("[%f  %f]",tau[i].x-test_tau[i].x,tau[i].y-test_tau[i].y);

			}
#endif			
            }else{
			fprintf(stderr,"[%s]:[%d] argv[4] should be based or batched or streamed !\n",__FUNCTION__,__LINE__);
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
