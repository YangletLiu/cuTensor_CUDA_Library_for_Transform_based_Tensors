#include "svd.h"
#include "based.h"
void streamedtsvd(float* t,const int m,const int n,const int tupe,float* U,float* S,float*  V){
	int ht  = tupe/2+1;
	int bat = m*n;
	float* d_t;
	cufftComplex* d_fftData;
	cudaMalloc((void**)&d_t,sizeof(float)*bat*tupe);
	cudaMalloc((void**)&d_fftData,sizeof(cufftComplex)*bat*ht);
	cudaMemcpy(d_t,t,sizeof(float)*bat*tupe,cudaMemcpyHostToDevice);

	//tff
	cufftHandle plan;
	int n_f[1]   = {tupe};
	int stride = bat,dist = 1;
	int in[1]  = {tupe};
	int on[1]  = {ht};
	size_t worksize=0;
	if (cufftPlanMany(&plan,1,n_f,in,stride,dist,on,stride,dist,
				CUFFT_R2C,bat)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	//estimate of the work size
/*	if(cufftGetSizeMany(plan,1,n_f,in,stride,dist,on,stride,dist,
			CUFFT_R2C,bat,&worksize)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Estimate work size failed!",__FUNCTION__,__LINE__);
		return;
 	}
	printf("the work size is:%lf G\n",(double)worksize/(1024*1024*1024));
*/
	if(cufftExecR2C(plan,d_t,(cufftComplex*)d_fftData)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	cudaFree(d_t);
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]cufftDestory faile!",__FUNCTION__,__LINE__);
		return;
	}
    //set stream for t
    cudaStream_t* stream = (cudaStream_t*)malloc(PLAN1D_SIZE*sizeof(cudaStream_t));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamCreate(&stream[i]);
    }

#if 1//tsvd
	cusolverDnHandle_t* handle=(cusolverDnHandle_t*)malloc(PLAN1D_SIZE*sizeof(cusolverDnHandle_t));
	gesvdjInfo_t* params=(gesvdjInfo_t*)malloc(ht*sizeof(gesvdjInfo_t));
	int* info = NULL;
	int echo = 1;
	int lda = m;
	int ldu = m;
	int ldv = n;
	int* lwork = (int*)malloc(ht*sizeof(int));
	cuComplex** work=NULL;

	//malloc u s v

	float* d_s = NULL;
	cuComplex* d_u = NULL;
	cuComplex* d_v = NULL;
	cudaMalloc((void**)&d_s,sizeof(float)*ht*((m<n)?m:n));
	cudaMalloc((void**)&d_u,sizeof(cuComplex)*ht*m*((m<n)?m:n));
	cudaMalloc((void**)&d_v,sizeof(cuComplex)*ht*n*((m<n)?m:n));
	cudaMalloc((void**)&info,sizeof(int)*ht);	
    //set stream
    for(int i=0;i<PLAN1D_SIZE;i++){	
	    if(cusolverDnCreate(&handle[i]) != CUSOLVER_STATUS_SUCCESS){
		    fprintf(stdout,"[%s]:[%d] cusolverDnCreate failed!",__FUNCTION__,__LINE__);
		    return;
    	}
        if(cusolverDnSetStream(handle[i],stream[i]) != CUSOLVER_STATUS_SUCCESS){
		    fprintf(stdout,"[%s]:[%d] cusolverDnCreate failed!",__FUNCTION__,__LINE__);
		    return;
        }
    
	}
    #pragma unroll
    for(int i=0;i<ht;i++){	
	if(cusolverDnCreateGesvdjInfo(&params[i]) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:creation svd info error",__FUNCTION__,__LINE__);
		return;
	}
    }	
    int tupe_num=ht/PLAN1D_SIZE;
    int tupe_s=ht%PLAN1D_SIZE;
    if(tupe_num > 0){
    for(int j=0;j<tupe_num;j++){
	for(int i=0;i<PLAN1D_SIZE;i++){
	if(cusolverDnCgesvdj_bufferSize(
			handle[i],
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_fftData,
			m,
			d_s,
			d_u,
			ldu,
			d_v,
			ldv,
			&lwork[i+j*PLAN1D_SIZE],
			params[i+j*PLAN1D_SIZE]) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR: create buffersize failed!",__FUNCTION__,__LINE__);
		return;
	        }
        }
    }
	for(int i=0;i<tupe_s;i++){
	if(cusolverDnCgesvdj_bufferSize(
			handle[i],
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_fftData,
			m,
			d_s,
			d_u,
			ldu,
			d_v,
			ldv,
			&lwork[i+tupe_num*PLAN1D_SIZE],
			params[i+tupe_num*PLAN1D_SIZE]) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR: create buffersize failed!",__FUNCTION__,__LINE__);
		return;
	        }
        }
    }else{
	for(int i=0;i<tupe_s;i++){
	if(cusolverDnCgesvdj_bufferSize(
			handle[i],
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_fftData,
			m,
			d_s,
			d_u,
			ldu,
			d_v,
			ldv,
			&lwork[i],
			params[i]) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR: create buffersize failed!",__FUNCTION__,__LINE__);
		return;
	        }
        }
    }
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
    work=(cuComplex**)malloc(ht*sizeof(cuComplex*));
    for(int i=0;i<ht;i++){
	if(cudaMalloc((void**)&work[i],sizeof(cuComplex)*lwork[i]) !=cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cudaMalloc error!",__FUNCTION__,__LINE__);
		return;
    }
    }
   /*for(int i=0;i<tupe;i++){
	if(cudaMalloc((void**)&work[i],sizeof(cuComplex)*lwork[i]) !=cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cudaMalloc error!",__FUNCTION__,__LINE__);
		return;
    }*/
	int step_d = m*n;
	int step_u = m*((m<n)?m:n);
	int step_s = ((m<n)?m:n);
	int step_v = n*((m<n)?m:n);	
    
    if(tupe_num >0){
        for(int j=0;j<tupe_num;j++){
	    for(int i=0;i<PLAN1D_SIZE;i++){
	    if(cusolverDnCgesvdj(
			handle[i],
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_fftData+step_d*i+j*step_d*PLAN1D_SIZE,
			lda,
			d_s+i*step_s+j*step_s*PLAN1D_SIZE,
			d_u+i*step_u+j*step_u*PLAN1D_SIZE,
			ldu,
			d_v+i*step_v+j*step_v*PLAN1D_SIZE,
			ldv,
			work[i+j*PLAN1D_SIZE],
			lwork[i+j*PLAN1D_SIZE],
			&info[i+j*PLAN1D_SIZE],
			params[i+j*PLAN1D_SIZE]) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:cusolverDnCgesvdj failed!",__FUNCTION__,__LINE__);
		return;
		    }   
	    }
        }
	    for(int i=0;i<tupe_s;i++){
	    if(cusolverDnCgesvdj(
			handle[i],
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_fftData+step_d*i+tupe_num*step_d*PLAN1D_SIZE,
			lda,
			d_s+i*step_s+tupe_num*step_s*PLAN1D_SIZE,
			d_u+i*step_u+tupe_num*step_u*PLAN1D_SIZE,
			ldu,
			d_v+i*step_v+tupe_num*step_v*PLAN1D_SIZE,
			ldv,
			work[i+tupe_num*PLAN1D_SIZE],
			lwork[i+tupe_num*PLAN1D_SIZE],
			&info[i+tupe_num*PLAN1D_SIZE],
			params[i+tupe_num*PLAN1D_SIZE]) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:cusolverDnCgesvdj failed!",__FUNCTION__,__LINE__);
		return;
		    }   
	    }
        
    }else{
	    for(int i=0;i<tupe_s;i++){
	    if(cusolverDnCgesvdj(
			handle[i],
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_fftData+step_d*i,
			lda,
			d_s+i*step_s,
			d_u+i*step_u,
			ldu,
			d_v+i*step_v,
			ldv,
			work[i],
			lwork[i],
			&info[i],
			params[i]) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:cusolverDnCgesvdj failed!",__FUNCTION__,__LINE__);
		return;
		    }
	    }
    }
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cusolverDnDestroy(handle[i])!=CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);
		return;
	    }
    }

    for(int i=0;i<ht;i++){
	if(cusolverDnDestroyGesvdjInfo(params[i])!=CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
    }
	
	if(d_fftData != NULL){
	cudaFree(d_fftData);
	d_fftData = NULL;
	}
	if(work != NULL){
    for(int i=0;i<tupe;i++){
        cudaFree(work[i]);
    }
	cudaFree(work);
	work = NULL;
	}
	if(info != NULL){
	cudaFree(info);
	info = NULL;
	}
#endif

	for(int i=0;i<PLAN1D_SIZE;i++){
    if(cudaStreamDestroy(stream[i]) != cudaSuccess){
        	fprintf(stdout,"[%s]:[%d] destory stream error!",__FUNCTION__,__LINE__);
        	return;	
    }
    }
    //ifft_u
	int threads=0;
	int blocks=0;
	
	cufftHandle iplan =0;
	in[0] = ht;
	on[0] = tupe;
	bat = m*((m<n)?m:n);
	stride = bat;
	float* du;
	cudaMalloc((void**)&du,sizeof(float)*bat*tupe);
	
	if (cufftPlanMany(&iplan,1,n_f,in,stride,dist,on,stride,dist,
				CUFFT_C2R,bat)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	//estimate of the work size
	if(cufftGetSizeMany(iplan,1,n_f,in,stride,dist,on,stride,dist,
			CUFFT_C2R,bat,&worksize)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Estimate work size failed!",__FUNCTION__,__LINE__);
		return;
 	}
	//printf("the work size is:%ld G\n",(double)worksize/(1024*1024*1024));

	if(cufftExecC2R(iplan,(cufftComplex*)d_u,du)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	int num=0;

	num=bat*tupe;
        if(num<512){
          threads=num;
          blocks=1;
        }else{
	  threads=512;
	  blocks=((num%512 ==0)?num/512:num/512+1);
	}
         fftResultProcess<<<blocks,threads>>>(du,num,tupe);
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	cudaMemcpy(U,du,sizeof(float)*bat*tupe,cudaMemcpyDeviceToHost);
	
	cudaFree(du);
	cudaFree(d_u);

	//ifft_v
	
	in[0] = ht;
	on[0] = tupe;
	bat = n*((m<n)?m:n);
	stride = bat;
	
	float* dv;
	cudaMalloc((void**)&dv,sizeof(float)*bat*tupe);
	if (cufftPlanMany(&iplan,1,n_f,in,stride,dist,on,stride,dist,
				CUFFT_C2R,bat)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	//estimate of the work size
	if(cufftGetSizeMany(iplan,1,n_f,in,stride,dist,on,stride,dist,
			CUFFT_C2R,bat,&worksize)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Estimate work size failed!",__FUNCTION__,__LINE__);
		return;
 	}
	//printf("the work size is:%ld G\n",(double)worksize/(1024*1024*1024));

	if(cufftExecC2R(iplan,(cufftComplex*)d_v,dv)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	num=bat*tupe;
        if(num<512){
          threads=num;
          blocks=1;
        }else{
	  threads=512;
	  blocks=((num%512 ==0)?num/512:num/512+1);
	}
         fftResultProcess<<<blocks,threads>>>(dv,num,tupe);
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	cudaMemcpy(V,dv,sizeof(float)*bat*tupe,cudaMemcpyDeviceToHost);
	
	cudaFree(dv);
	cudaFree(d_v);
	//ifft_s

	bat = ((m<n)?m:n);
	stride = bat;

	cuComplex* d_s2;
	cudaMalloc((void**)&d_s2,sizeof(cuComplex)*ht*bat);
	float* d_s3;
	cudaMalloc((void**)&d_s3,sizeof(float)*tupe*bat);
		
	num=bat*ht;
	if(ht*bat<512){
	        threads=num;	
       		blocks=1;
     	}else{
	        threads=512;
	        blocks=((num%512 ==0)?num/512:num/512+1);
	}

    	float2cuComplex<<<blocks,threads>>>(d_s,ht*bat,d_s2);
	
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	cudaFree(d_s);

	
	if (cufftPlanMany(&iplan,1,n_f,in,stride,dist,on,stride,dist,
				CUFFT_C2R,bat)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	if(cufftExecC2R(iplan,(cufftComplex*)d_s2,d_s3) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
       num=bat*tupe;
       if(num<512){
          threads=num;
          blocks=1;
       }else{
	  threads=512;
	  blocks=((num%512 ==0)?num/512:num/512+1);
	}
         fftResultProcess<<<blocks,threads>>>(d_s3,bat*tupe,tupe);
	
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaMemcpy(S,d_s3,sizeof(float)*tupe*bat,cudaMemcpyDeviceToHost);


	if(cufftDestroy(iplan)!=CUFFT_SUCCESS){

		fprintf(stdout,"[%s]:[%d]cufftDestory failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	cudaFree(d_s3);
	cudaFree(d_s2);
	
}
