#include "norm.h"
void batchedtnorm(float* t,const int m, const int n,const int tube,float* result){
   int num=m*n*tube;
   float* y;
   cudaMalloc((void**)&y,num*sizeof(float));
   //set vector
   if(cublasSetVector(num,sizeof(float),t,1,y,1) != CUBLAS_STATUS_SUCCESS){
        fprintf(stdout,"[%s]:[%d] cublasSnrm2 error!",__FUNCTION__,__LINE__);
        return;
   }
   //create cublasHandle;
   cublasHandle_t handle;
   cublasCreate(&handle);
   //norm
   if(cublasSnrm2(handle,num,y,1,result) != CUBLAS_STATUS_SUCCESS){
        fprintf(stdout,"[%s]:[%d] cublasSnrm2 error!",__FUNCTION__,__LINE__);
        return;
   }
   
   if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS){
        fprintf(stdout,"[%s]:[%d] cublasSnrm2 error!",__FUNCTION__,__LINE__);
        return;
   }
   if(y != NULL){
        cudaFree(y);
        y=NULL;
   }    
}
