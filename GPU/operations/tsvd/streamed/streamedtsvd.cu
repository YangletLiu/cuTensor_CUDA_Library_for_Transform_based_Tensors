#include "svd.h"
#include "based.h"
void streamedtsvd(float* t,const int m,const int n,const int tupe,float* U,float* S,float*  V){
    int bat =m*n;
    cufftComplex* t_f;

    cudaHostAlloc((void**)&t_f,bat*tupe*sizeof(cufftComplex),cudaHostAllocDefault);
    
    //transform t1
	for(int i=0;i<bat;i++){
	   for(int j=0;j<tupe;j++){
		t_f[i*tupe+j].x=t[j*bat+i];
		t_f[i*tupe+j].y=0;
		}
	}
/*printf("\n============================\n");
for(int i=0;i<bat*tupe;i++){
    printf("[%f %f]",t_f[i].x,t_f[i].y);
}	
printf("\n============================\n");
*/
    //set stream for t
    cudaStream_t* stream = (cudaStream_t*)malloc(PLAN1D_SIZE*sizeof(cudaStream_t));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamCreate(&stream[i]);
    }

	//tfft:C2C
	cufftComplex* d_fftData;
	cudaMalloc((void**)&d_fftData,tupe*bat*sizeof(cufftComplex));	
    //process bat
    int bat_num = bat/PLAN1D_SIZE;
    int bat_s = bat%PLAN1D_SIZE;
	cufftHandle * plan=(cufftHandle*)malloc(sizeof(cufftHandle)*PLAN1D_SIZE);
    memset(plan,0,sizeof(cufftHandle));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftPlan1d(&plan[i],tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
	 	fprintf(stdout,"[%s]:[%d] cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;
	}
        cufftSetStream(plan[i],stream[i]);
    }
    if(bat_num > 0){
    for(int j=0;j<bat_num;j++){

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,t_f+i*tupe+j*tupe*PLAN1D_SIZE,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]);
     }

    #pragma unroll	
	for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(plan[i],d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
            	}
        	}
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(t_f+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
         }
    }

    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,t_f+i*tupe+bat_num*tupe*PLAN1D_SIZE,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]);
    }
    #pragma unroll	
	for(int i=0;i<bat_s;i++){
	if(cufftExecC2C(plan[i],d_fftData+i*tupe+bat_num*tupe*PLAN1D_SIZE,d_fftData+i*tupe+PLAN1D_SIZE*bat_num*tupe,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(t_f+i*tupe+bat_num*tupe*PLAN1D_SIZE,d_fftData+i*tupe+tupe*bat_num*PLAN1D_SIZE,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
    }
    }else{
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(d_fftData+i*tupe,t_f+i*tupe,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]);
    }
    #pragma unroll	
	for(int i=0;i<bat_s;i++){
	if(cufftExecC2C(plan[i],d_fftData+i*tupe,d_fftData+i*tupe,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	    }
	}
    #pragma unroll
    for(int i=0;i<bat_s;i++){
	cudaMemcpyAsync(t_f+i*tupe,d_fftData+i*tupe,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
    }
    }
	//transform
/*printf("\n============================\n");
for(int i=0;i<bat*tupe;i++){
    printf("[%f %f]",t_f[i].x,t_f[i].y);
}	
printf("\n============================\n");
*/
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
    cufftComplex* t_f2 = (cufftComplex*)malloc(sizeof(cufftComplex)*tupe*bat);
	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		t_f2[j*bat+i]=t_f[i*tupe+j];
		}
	}
/*printf("\n============================\n");
for(int i=0;i<bat*tupe;i++){
    printf("[%f %f]",t_f2[i].x,t_f2[i].y);
}	
printf("\n============================\n");
*/	cudaMemcpy(d_fftData,t_f2,sizeof(cufftComplex)*bat*tupe,cudaMemcpyHostToDevice);

    for(int i=0;i<PLAN1D_SIZE;i++){	
	    if(cufftDestroy(plan[i])!=CUFFT_SUCCESS){
		    fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		    return;
	    }
    }
		
	if(t_f != NULL){
	cudaFreeHost(t_f);
	t_f = NULL;
	}
	if(t_f2 !=NULL){
	free(t_f2);
	t_f2 = NULL;	
	}
#if 1//tsvd
	cusolverDnHandle_t* handle=(cusolverDnHandle_t*)malloc(PLAN1D_SIZE*sizeof(cusolverDnHandle_t));
	gesvdjInfo_t* params=(gesvdjInfo_t*)malloc(tupe*sizeof(gesvdjInfo_t));
	int* info = NULL;
	int echo = 1;
	int lda = m;
	int ldu = m;
	int ldv = n;
	int* lwork = (int*)malloc(tupe*sizeof(int));
	cuComplex** work=NULL;

	//malloc u s v

	float* d_s = NULL;
	cuComplex* d_u = NULL;
	cuComplex* d_v = NULL;
	cudaMalloc((void**)&d_s,sizeof(float)*tupe*((m<n)?m:n));
	cudaMalloc((void**)&d_u,sizeof(cuComplex)*tupe*m*((m<n)?m:n));
	cudaMalloc((void**)&d_v,sizeof(cuComplex)*tupe*n*((m<n)?m:n));
	cudaMalloc((void**)&info,sizeof(int)*tupe);	
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
    for(int i=0;i<tupe;i++){	
	if(cusolverDnCreateGesvdjInfo(&params[i]) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:creation svd info error",__FUNCTION__,__LINE__);
		return;
	}
    }	
    int tupe_num=tupe/PLAN1D_SIZE;
    int tupe_s=tupe%PLAN1D_SIZE;
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
    work=(cuComplex**)malloc(tupe*sizeof(cuComplex*));
    for(int i=0;i<tupe;i++){
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

    for(int i=0;i<tupe;i++){
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
	//ifft
	//transform
//	cuComplex* h_u = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_u);
    cuComplex* h_u;
    cudaHostAlloc((void**)&h_u,tupe*step_u*sizeof(cuComplex),cudaHostAllocDefault);
    cuComplex* h_u2 = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_u);

//	cuComplex* h_v = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_v);
    cuComplex* h_v;
    cudaHostAlloc((void**)&h_v,tupe*step_v*sizeof(cuComplex),cudaHostAllocDefault);
    cuComplex* h_v2 = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_v);

//	cuComplex* h_s = (cuComplex*)malloc(sizeof(cuComplex)*tupe*step_s);
    cuComplex* h_s;
    cudaHostAlloc((void**)&h_s,tupe*step_s*sizeof(cuComplex),cudaHostAllocDefault);
    float* h_s2 = (float*)malloc(sizeof(float)*tupe*step_s);

	cuComplex* d_s2;
	cudaMalloc((void**)&d_s2,sizeof(cuComplex)*tupe*step_s);

	cudaMemcpy(h_u2,d_u,sizeof(cuComplex)*tupe*step_u,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v2,d_v,sizeof(cuComplex)*tupe*step_v,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_s2,d_s,sizeof(float)*tupe*step_s,cudaMemcpyDeviceToHost);
	
/*printf("\n============================\n");
for(int i=0;i<step_s*tupe;i++){
    printf("[%f ]",h_s2[i]);
}	
printf("\n============================\n");
*/	//transform_u
	for(int i=0;i<step_u;i++){
	  for(int j=0;j<tupe;j++){
		h_u[i*tupe+j]=h_u2[j*step_u+i];
		}
	}

	//transform_v
	for(int i=0;i<step_v;i++){
	  for(int j=0;j<tupe;j++){
		h_v[i*tupe+j]=h_v2[j*step_v+i];
		}
	}
		
	//transform_s
	for(int i=0;i<step_s;i++){
	  for(int j=0;j<tupe;j++){
		h_s[i*tupe+j].x=h_s2[j*step_s+i];
		h_s[i*tupe+j].y=0;
		}
	}
	
/*	for(int i=0;i<tupe*step_s;i++){
		printf("%f ",h_s2[i]);
	}
	printf("\n");
*/
	
	if(h_u2 != NULL){
	free(h_u2);
	h_u2 = NULL;
	}
	if(h_v2 != NULL){
	free(h_v2);
	h_v2 = NULL;
	}
	if(h_s2 != NULL){
	free(h_s2);
	h_s2= NULL;
	}
	if(d_s != NULL){
	cudaFree(d_s);
	d_s = NULL;
	}

	cufftHandle* iplan =(cufftHandle*)malloc(PLAN1D_SIZE*sizeof(cufftHandle));
    for(int i=0;i<PLAN1D_SIZE;i++){

	if(cufftPlan1d(&iplan[i],tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
		    fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		    return;	
	}

    if(cufftSetStream(iplan[i],stream[i]) != CUFFT_SUCCESS){            
		    fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		    return;	
        }
	}

//	cudaMemcpy(d_s2,h_s,sizeof(cuComplex)*tupe*step_s,cudaMemcpyHostToDevice);
//	cudaMemcpy(d_v,h_v,sizeof(cuComplex)*tupe*step_v,cudaMemcpyHostToDevice);
	//ifft_u
    int step_u_num=step_u/PLAN1D_SIZE;
    int step_u_s=step_u%PLAN1D_SIZE;
    if(step_u_num > 0){
    for(int j=0;j<step_u_num;j++){
    for(int i=0;i<PLAN1D_SIZE;i++){
    if(cudaMemcpyAsync(d_u+i*tupe+j*tupe*PLAN1D_SIZE,h_u+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
                }
	}
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(iplan[i],d_u+i*tupe+j*tupe*PLAN1D_SIZE,d_u+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cudaMemcpyAsync(h_u+i*tupe+j*tupe*PLAN1D_SIZE,d_u+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
        }
    }
    }
    for(int i=0;i<step_u_s;i++){
    if(cudaMemcpyAsync(d_u+i*tupe+step_u_num*tupe*PLAN1D_SIZE,h_u+i*tupe+step_u_num*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
                }
	}
    for(int i=0;i<step_u_s;i++){
	if(cufftExecC2C(iplan[i],d_u+i*tupe+step_u_num*tupe*PLAN1D_SIZE,d_u+i*tupe+step_u_num*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
    for(int i=0;i<step_u_s;i++){
	if(cudaMemcpyAsync(h_u+i*tupe+step_u_num*tupe*PLAN1D_SIZE,d_u+i*tupe+step_u_num*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
        }
    }
    }else{
    for(int i=0;i<step_u_s;i++){
    if(cudaMemcpyAsync(d_u+i*tupe,h_u+i*tupe,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
                }
	}
    for(int i=0;i<step_u_s;i++){
	if(cufftExecC2C(iplan[i],d_u+i*tupe,d_u+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
    for(int i=0;i<step_u_s;i++){
	if(cudaMemcpyAsync(h_u+i*tupe,d_u+i*tupe,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
        }
    }
    }
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
	//ifft_v
    int step_v_num=step_v/PLAN1D_SIZE;
    int step_v_s=step_v%PLAN1D_SIZE;
    if(step_v_num > 0){
    for(int j=0;j<step_v_num;j++){
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cudaMemcpyAsync(d_v+i*tupe+j*tupe*PLAN1D_SIZE,h_v+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) !=cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }       
	    }
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(iplan[i],d_v+i*tupe+j*tupe*PLAN1D_SIZE,d_v+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]CUFFT ERROR: cufftExecc2Cfailed!",__FUNCTION__,__LINE__);
		return;
	        }
	    }
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cudaMemcpyAsync(h_v+i*tupe+j*tupe*PLAN1D_SIZE,d_v+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
        }
    }
    for(int i=0;i<step_v_s;i++){
	if(cudaMemcpyAsync(d_v+i*tupe+step_v_num*tupe*PLAN1D_SIZE,h_v+i*tupe+step_v_num*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) !=cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }       
	    }
    for(int i=0;i<step_v_s;i++){
	if(cufftExecC2C(iplan[i],d_v+i*tupe+step_v_num*tupe*PLAN1D_SIZE,d_v+i*tupe+step_v_num*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]CUFFT ERROR: cufftExecc2Cfailed!",__FUNCTION__,__LINE__);
		return;
	        }
	    }
    for(int i=0;i<step_v_s;i++){
	if(cudaMemcpyAsync(h_v+i*tupe+step_v_num*tupe*PLAN1D_SIZE,d_v+i*tupe+step_v_num*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
        }
    }else{
    for(int i=0;i<step_v_s;i++){
	if(cudaMemcpyAsync(d_v+i*tupe,h_v+i*tupe,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) !=cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }       
	    }
    for(int i=0;i<step_v_s;i++){
	if(cufftExecC2C(iplan[i],d_v+i*tupe,d_v+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]CUFFT ERROR: cufftExecc2Cfailed!",__FUNCTION__,__LINE__);
		return;
	        }
	    }
    for(int i=0;i<step_v_s;i++){
	if(cudaMemcpyAsync(h_v+i*tupe,d_v+i*tupe,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
        }
    }
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
	
	//ifft_s
    int step_s_num=step_s/PLAN1D_SIZE;
    int step_s_s=step_s%PLAN1D_SIZE;
    if(step_s_num > 0){
    for(int j=0;j<step_s_num;j++){
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cudaMemcpyAsync(d_s2+i*tupe+j*tupe*PLAN1D_SIZE,h_s+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) !=cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
	    }
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(iplan[i],d_s2+i*tupe+j*tupe*PLAN1D_SIZE,d_s2+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	
	        }
	    }
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cudaMemcpyAsync(h_s+i*tupe+j*tupe*PLAN1D_SIZE,d_s2+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
	    }
       }
    for(int i=0;i<step_s_s;i++){
	if(cudaMemcpyAsync(d_s2+i*tupe+step_s_num*tupe*PLAN1D_SIZE,h_s+i*tupe+step_s_num*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) !=cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
	    }
    for(int i=0;i<step_s_s;i++){
	if(cufftExecC2C(iplan[i],d_s2+i*tupe+step_s_num*tupe*PLAN1D_SIZE,d_s2+i*tupe+step_s_num*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	
	        }
	    }
    for(int i=0;i<step_s_s;i++){
	if(cudaMemcpyAsync(h_s+i*tupe+step_s_num*tupe*PLAN1D_SIZE,d_s2+i*tupe+step_s_num*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
    }
    }else{
    for(int i=0;i<step_s_s;i++){
	if(cudaMemcpyAsync(d_s2+i*tupe,h_s+i*tupe,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]) !=cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
	    }
    for(int i=0;i<step_s_s;i++){
	if(cufftExecC2C(iplan[i],d_s2+i*tupe,d_s2+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	
	        }
	    }
    for(int i=0;i<step_s_s;i++){
	if(cudaMemcpyAsync(h_s+i*tupe,d_s2+i*tupe,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]) != cudaSuccess){
		        fprintf(stdout,"[%s]:[%d] cudaMemcpyAsync error!",__FUNCTION__,__LINE__);
	    	    return;
            }
    }
    }
    for(int i=0;i<PLAN1D_SIZE;i++){
        if(cudaStreamSynchronize(stream[i]) !=cudaSuccess){
        	fprintf(stdout,"[%s]:[%d] cudaStreamSynchronize error!",__FUNCTION__,__LINE__);
        	return;	
            }
        }
	//transform_u
	for(int i=0;i<step_u;i++){
	  for(int j=0;j<tupe;j++){
		U[j*step_u+i]=h_u[i*tupe+j].x/tupe;
	//	U[j*step_u+i].y=h_u[i*tupe+j].y/tupe;
		}
	}

	//transform_v
	for(int i=0;i<step_v;i++){
	  for(int j=0;j<tupe;j++){
		V[j*step_v+i]=h_v[i*tupe+j].x/tupe;
	//	V[j*step_v+i].y=h_v[i*tupe+j].y/tupe;
		}
	}	

	//transform_s
	for(int i=0;i<step_s;i++){
	  for(int j=0;j<tupe;j++){
		S[j*step_s+i]=h_s[i*tupe+j].x/tupe;
	//	S[j*step_s+j].y=h_s[i*tupe+j].y/tupe;
		}
	}

	for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftDestroy(iplan[i])!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
    if(cudaStreamDestroy(stream[i]) != cudaSuccess){
        	fprintf(stdout,"[%s]:[%d] destory stream error!",__FUNCTION__,__LINE__);
        	return;	
    }
    }

	if(d_u != NULL){	
	cudaFree(d_u);
	d_u =NULL;
	}
	if(d_v != NULL){
	cudaFree(d_v);
	d_v = NULL;
	}
	if(d_s2 != NULL){
	cudaFree(d_s2);
	d_s2 = NULL;
	}
	if(h_u !=NULL){
	cudaFreeHost(h_u);
	h_u = NULL;
	}
	if(h_v != NULL){
	cudaFreeHost(h_v);
	h_v = NULL;
	}
	if(h_s != NULL){
	cudaFreeHost(h_s);
	h_s = NULL;
    }
}
