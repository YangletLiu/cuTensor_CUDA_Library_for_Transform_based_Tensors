#include "fft.h"
void streamedTfft(float *t,int l,int bat,cufftComplex *tf) {
    cufftComplex *t_f;
    if( cudaHostAlloc((void**)&t_f,sizeof(cufftComplex)*l*bat,cudaHostAllocDefault) != cudaSuccess){
        	fprintf(stdout,"[%s]:[%d] cudaHostAlloc error!",__FUNCTION__,__LINE__);
        	return;	
    }
    //transform
    for(int i=0;i<bat;i++)
      for(int j=0;j<l;j++){
        t_f[i*l+j].x=t[j*bat+i];
        t_f[i*l+j].y=0;
      }
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));

    //set stream
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t)*MAX_PLAN1D_SIZE);
    cufftHandle* plan = (cufftHandle*)malloc(sizeof(cufftHandle)*MAX_PLAN1D_SIZE);
    memset(plan,0,sizeof(cufftHandle));
    
    int bat_num=bat/MAX_PLAN1D_SIZE;
    int s_bat=bat%MAX_PLAN1D_SIZE;  
//  
    #pragma unroll
    for(int i=0;i<MAX_PLAN1D_SIZE;i++){
       if( cudaStreamCreate(&stream[i]) != cudaSuccess){
        	fprintf(stdout,"[%s]:[%d] create stream error!",__FUNCTION__,__LINE__);
        	return;	
       }
       }
    #pragma unroll
    for(int i=0;i<MAX_PLAN1D_SIZE;i++){
        if(cufftPlan1d(&plan[i],l,CUFFT_C2C, 1)!=CUFFT_SUCCESS){
        	fprintf(stdout,"[%s]:[%d] fft cufftPlan1d error!",__FUNCTION__,__LINE__);
        	return;	
    	}
       if(cufftSetStream(plan[i], stream[i]) != CUFFT_SUCCESS){
        	fprintf(stdout,"[%s]:[%d] fft set stream error!",__FUNCTION__,__LINE__);
        	return;	
       }
    }
   if(bat_num > 0){
  //  int j=0;
    for(int j=0;j<bat_num;j++){
    #pragma unroll
    for(int i=0;i<MAX_PLAN1D_SIZE;i++){
       if( cudaMemcpyAsync(d_fftData+i*l+j*MAX_PLAN1D_SIZE*l,t_f+i*l+j*MAX_PLAN1D_SIZE*l,l*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
            fprintf(stdout,"[%s]:[%d] cudaMencpyAsync error!",__FUNCTION__,__LINE__);
            return;
        }
        if(cufftExecC2C(plan[i],(cufftComplex*)(d_fftData+i*l+j*MAX_PLAN1D_SIZE*l),(cufftComplex*)(d_fftData+i*l+j*MAX_PLAN1D_SIZE*l),CUFFT_FORWARD)!=CUFFT_SUCCESS){
        	fprintf(stdout,"[%s]:[%d] fft cufftExecC2c error!",__FUNCTION__,__LINE__);
         	return;
    	}

        cudaMemcpyAsync(t_f+i*l+j*MAX_PLAN1D_SIZE*l,d_fftData+i*l+j*MAX_PLAN1D_SIZE*l,l*sizeof(cufftComplex),cudaMemcpyDeviceToHost,stream[i]);

    }
    }

    
    #pragma unroll
    for(int i=0;i<s_bat;i++){
         if( cudaMemcpyAsync(d_fftData+i*l+bat_num*MAX_PLAN1D_SIZE*l,t_f+i*l+bat_num*MAX_PLAN1D_SIZE*l,l*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
            fprintf(stdout,"[%s]:[%d] cudaMencpyAsync error!",__FUNCTION__,__LINE__);
            return;
        }

        if(cufftExecC2C(plan[i],(cufftComplex*)(d_fftData+i*l+bat_num*MAX_PLAN1D_SIZE*l),(cufftComplex*)(d_fftData+i*l+bat_num*MAX_PLAN1D_SIZE*l),CUFFT_FORWARD)!=CUFFT_SUCCESS){
        	fprintf(stdout,"[%s]:[%d] fft cufftExecC2c error!",__FUNCTION__,__LINE__);
         	return;
    	}

         cudaMemcpyAsync(t_f+i*l+bat_num*MAX_PLAN1D_SIZE*l,d_fftData+i*l+bat_num*MAX_PLAN1D_SIZE*l,l*sizeof(cufftComplex),cudaMemcpyDeviceToHost,stream[i]);
         }

    }else{
    #pragma unroll
    for(int i=0;i<s_bat;i++){
         if( cudaMemcpyAsync(d_fftData+i*l,t_f+i*l,l*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
            fprintf(stdout,"[%s]:[%d] cudaMencpyAsync error!",__FUNCTION__,__LINE__);
            return;
        }

        if(cufftExecC2C(plan[i],(cufftComplex*)(d_fftData+i*l),(cufftComplex*)(d_fftData+i*l),CUFFT_FORWARD)!=CUFFT_SUCCESS){
        	fprintf(stdout,"[%s]:[%d] fft cufftExecC2c error!",__FUNCTION__,__LINE__);
         	return;
    	}

         cudaMemcpyAsync(t_f+i*l,d_fftData+i*l,l*sizeof(cufftComplex),cudaMemcpyDeviceToHost,stream[i]);
         }
    }
    //synchronize stream

    #pragma unroll
    for(int i=0;i<MAX_PLAN1D_SIZE;i++){
    cudaStreamSynchronize(stream[i]);
    }

    //destroy stream
   
   #pragma unroll
    for (int i=0; i<MAX_PLAN1D_SIZE; i++){
        cufftDestroy(plan[i]);
        cudaStreamDestroy(stream[i]);
    }
        free(plan);
        free(stream);
        cudaFree(d_fftData);
    
    //transform

    for(int i=0;i<bat;i++)
          for(int j=0;j<l;j++){
            tf[j*bat+i]=t_f[i*l+j];
          }
    cudaFreeHost(t_f);
}

void streamedTifft(float *t, int l, int bat, cufftComplex *tf){
    cufftComplex *t_f;
    cudaHostAlloc((void**)&t_f,l*bat*sizeof(cufftComplex),cudaHostAllocDefault);
    //transform
    for(int i=0;i<bat;i++)
      for(int j=0;j<l;j++){
        t_f[i*l+j]=tf[j*bat+i];
      }
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));
    
    cudaStream_t * stream = (cudaStream_t *)malloc(MAX_PLAN1D_SIZE*sizeof(cudaStream_t));
    cufftHandle* plan = (cufftHandle*)malloc(MAX_PLAN1D_SIZE*sizeof(cufftHandle));
    memset(plan,0,sizeof(cufftHandle)); 

    #pragma unroll
    for(int i=0;i<MAX_PLAN1D_SIZE;i++){
        cudaStreamCreate(&stream[i]);
    }
    #pragma unroll
   for(int i=0;i<MAX_PLAN1D_SIZE;i++){
        if(cufftPlan1d(&plan[i],l,CUFFT_C2C,1)!=CUFFT_SUCCESS){
    	fprintf(stdout,"[%s]:[%d] fft cufftPlan1d error!",__FUNCTION__,__LINE__);
    	return;	
    	}
       if( cufftSetStream(plan[i],stream[i]) != CUFFT_SUCCESS){
    	fprintf(stdout,"[%s]:[%d] fft set stream error!",__FUNCTION__,__LINE__);
    	return;	
       }
    }
    int bat_num=bat/MAX_PLAN1D_SIZE;
    int s_bat=bat%MAX_PLAN1D_SIZE;  
    
    if(bat_num !=0){
    for(int j=0;j<bat_num;j++){
    #pragma unroll
    for(int i=0;i<MAX_PLAN1D_SIZE;i++){
         if( cudaMemcpyAsync(d_fftData+i*l+j*l*MAX_PLAN1D_SIZE,t_f+i*l+j*l*MAX_PLAN1D_SIZE,l*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
            fprintf(stdout,"[%s]:[%d] cudaMencpyAsync error!",__FUNCTION__,__LINE__);
            return;
        }

        if(cufftExecC2C(plan[i],(cufftComplex*)(d_fftData+i*l+j*l*MAX_PLAN1D_SIZE),(cufftComplex*)(d_fftData+i*l+j*l*MAX_PLAN1D_SIZE),CUFFT_INVERSE)!=CUFFT_SUCCESS){
        	fprintf(stdout,"[%s]:[%d] fft cufftExecC2c error!",__FUNCTION__,__LINE__);
         	return;
    	}

         cudaMemcpyAsync(t_f+i*l+j*l*MAX_PLAN1D_SIZE,d_fftData+i*l+j*l*MAX_PLAN1D_SIZE,l*sizeof(cufftComplex),cudaMemcpyDeviceToHost,stream[i]);
    }
    }
    #pragma unroll
    for(int i=0;i<s_bat;i++){
         if( cudaMemcpyAsync(d_fftData+i*l+bat_num*l*MAX_PLAN1D_SIZE,t_f+i*l+bat_num*l*MAX_PLAN1D_SIZE,l*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
            fprintf(stdout,"[%s]:[%d] cudaMencpyAsync error!",__FUNCTION__,__LINE__);
            return;
        }

        if(cufftExecC2C(plan[i],(cufftComplex*)(d_fftData+i*l+bat_num*l*MAX_PLAN1D_SIZE),(cufftComplex*)(d_fftData+i*l+MAX_PLAN1D_SIZE*l*bat_num),CUFFT_INVERSE)!=CUFFT_SUCCESS){
        	fprintf(stdout,"[%s]:[%d] fft cufftExecC2c error!",__FUNCTION__,__LINE__);
         	return;
    	}

         cudaMemcpyAsync(t_f+i*l+bat_num*MAX_PLAN1D_SIZE*l,d_fftData+i*l+bat_num*MAX_PLAN1D_SIZE*l,l*sizeof(cufftComplex),cudaMemcpyDeviceToHost,stream[i]);
         }

    }else{
    #pragma unroll
    for(int i=0;i<s_bat;i++){
         if( cudaMemcpyAsync(d_fftData+i*l,t_f+i*l,l*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
            fprintf(stdout,"[%s]:[%d] cudaMencpyAsync error!",__FUNCTION__,__LINE__);
            return;
        }

        if(cufftExecC2C(plan[i],(cufftComplex*)(d_fftData+i*l),(cufftComplex*)(d_fftData+i*l),CUFFT_INVERSE)!=CUFFT_SUCCESS){
        	fprintf(stdout,"[%s]:[%d] fft cufftExecC2c error!",__FUNCTION__,__LINE__);
         	return;
    	}

         cudaMemcpyAsync(t_f+i*l,d_fftData+i*l,l*sizeof(cufftComplex),cudaMemcpyDeviceToHost,stream[i]);
         }
    }
    //synchronize stream
   
    #pragma unroll
    for(int i=0;i<MAX_PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
    #pragma unroll
    for(int i=0;i<MAX_PLAN1D_SIZE;i++){
       cufftDestroy(plan[i]);
       cudaStreamDestroy(stream[i]);
    }
    free(plan);
    free(stream);
    cudaFree(d_fftData);
    //transform
    for(int i=0;i<bat;i++)
          for(int j=0;j<l;j++){
            t[j*bat+i]=t_f[i*l+j].x/l;
          }
    cudaFreeHost(t_f);
}
