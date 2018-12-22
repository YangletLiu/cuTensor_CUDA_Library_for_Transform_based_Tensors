#if 1
#include"assert.h"
#include<cuda_runtime.h>
#include"cusolverDn.h"
#include<time.h>
#include<stdio.h>
#include<stdlib.h>
void svd_complex(int m,int n,cuComplex* T,cuComplex* U,cuComplex* V,float* S);

int main(int argc,char* argv[]){
	int m;
	int n;
	m=atoi(argv[1]);
	n=atoi(argv[2]);

	
	cuComplex* T=(cuComplex*)malloc(sizeof(cuComplex)*m*n);
	cuComplex* u=(cuComplex*)malloc(sizeof(cuComplex)*m*((m<n)?m:n));
	cuComplex* v=(cuComplex*)malloc(sizeof(cuComplex)*n*((m<n)?m:n));
	float* s=(float*)malloc(sizeof(float)*((m<n)?m:n));
	
	for(int i=0;i<m*n;i++){
          T[i].x=(float)rand()/(RAND_MAX/100);
          T[i].y=(float)rand()/(RAND_MAX/100);
	}
         clock_t start,end;
  
         start = clock();
         svd_complex(m,n,T,u,v,s);
         end = clock();
  
        double time = (double)(end-start)/CLOCKS_PER_SEC;
        printf("%d %d %lf\n",m,n,time);
	return 1;
}

 void svd_complex(int m,int n,cuComplex* T,cuComplex* U,cuComplex* V,float* S){
     cusolverDnHandle_t handle;
     gesvdjInfo_t params=NULL;
     int* info=NULL;
     int echo=1;
     int lda=0;
     lda=m;
     int ldu=0;
     ldu=m;
     int ldv=0;
     ldv=n;
     int lwork=0;
     cuComplex* work=NULL;
     float* s=NULL;
     cuComplex* u=NULL;
     cuComplex* v=NULL;
     cuComplex* t=NULL;
     cusolverStatus_t status=CUSOLVER_STATUS_SUCCESS;
     status=cusolverDnCreate(&handle);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     status=cusolverDnCreateGesvdjInfo(&params);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     cudaError_t stat1=cudaSuccess;
     cudaError_t stat2=cudaSuccess;
     cudaError_t stat3=cudaSuccess;
     cudaError_t stat4=cudaSuccess;
     cudaError_t stat5=cudaSuccess;
     cudaError_t stat6=cudaSuccess;
     stat1=cudaMalloc((void**)&info,sizeof(int));
     int* inf=(int*)malloc(sizeof(int));
     stat2=cudaMalloc((void**)&u,sizeof(cuComplex)*m*((m<n)?m:n));
     stat3=cudaMalloc((void**)&v,sizeof(cuComplex)*n*((m<n)?m:n));
     stat4=cudaMalloc((void**)&s,sizeof(float)*((m<n)?m:n));
     stat5=cudaMalloc((void**)&t,sizeof(cuComplex)*m*n);
     stat6=cudaMemcpy(t,T,sizeof(cuComplex)*m*n,cudaMemcpyHostToDevice);
     if(
    		 stat1!=cudaSuccess||
    		 stat2!=cudaSuccess||
    		 stat3!=cudaSuccess||
    		 stat4!=cudaSuccess||
    		 stat5!=cudaSuccess||
    		 stat6!=cudaSuccess){
    	 printf("cuda malloc error\n");
    	 exit(-1);
     }
     if(cusolverDnCgesvdj_bufferSize(
    		 handle,
    		 CUSOLVER_EIG_MODE_VECTOR,
    		 echo,
    		 m,
    		 n,
    		 t,
    		 m,
    		 s,
    		 u,
    		 ldu,
    		 v,
    		 ldv,
    		 &lwork,
    		 params)!=CUSOLVER_STATUS_SUCCESS){
    	 printf("cusolverDnCgesvdj_bufferSize failed\n");
    	 exit(-1);

     }
     if(cudaDeviceSynchronize()!=cudaSuccess){
    	 printf("synchronize failed");
    	 exit(-1);
     }
     stat1=cudaMalloc((void**)&work,sizeof(cuComplex)*lwork);
     assert(stat1==cudaSuccess);
     if(cusolverDnCgesvdj(
    		 handle,
    		 CUSOLVER_EIG_MODE_VECTOR,
    		 echo,
    		 m,
    		 n,
    		 t,
    		 lda,
    		 s,
    		 u,
    		 ldu,
    		 v,
    		 ldv,
    		 work,
    		 lwork,
    		 info,
    		 params)!=CUSOLVER_STATUS_SUCCESS){
    	 printf("cusolverDnCgesvdj err\n");
    	 return;
     }
     if(cudaDeviceSynchronize()!=cudaSuccess){
    	 printf("cuda synchronize err\n");
    	 return;
     }
     stat1=cudaMemcpy(U,u,sizeof(cuComplex)*m*((m<n)?m:n),cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     stat1=cudaMemcpy(V,v,sizeof(cuComplex)*n*((m<n)?m:n),cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     stat1=cudaMemcpy(S,s,sizeof(float)*((m<n)?m:n),cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     cudaMemcpy(inf,info,sizeof(int),cudaMemcpyDeviceToHost);
     free(inf);
     stat1=cudaFree(u);
     assert(stat1==cudaSuccess);
     stat1=cudaFree(v);
     assert(stat1==cudaSuccess);
     stat1=cudaFree(s);
     assert(stat1==cudaSuccess);
     cudaFree(info);
     cudaFree(work);
     status=cusolverDnDestroy(handle);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     status=cusolverDnDestroyGesvdjInfo(params);
     assert(status==CUSOLVER_STATUS_SUCCESS);
}
#endif
