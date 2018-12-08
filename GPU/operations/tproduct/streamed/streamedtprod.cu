#include "tprod.h"
#include "based.h"
void streamedtprod(float* t1,float* t2,float* T,int row, int col, int rank, int tupe) {
	int bat = row*rank;
	int bat2 = rank*col;
//	cufftComplex* t_f = (cufftComplex*)malloc(bat*tupe*sizeof(cufftComplex));
//	cufftComplex* t_f2 = (cufftComplex*)malloc(bat2*tupe*sizeof(cufftComplex));
    cufftComplex* t_f;
    cufftComplex* t_f2;

    cudaHostAlloc((void**)&t_f,bat*tupe*sizeof(cufftComplex),cudaHostAllocDefault);
    cudaHostAlloc((void**)&t_f2,bat2*tupe*sizeof(cufftComplex),cudaHostAllocDefault);
    

    //transform t1
	for(int i=0;i<bat;i++){
	   for(int j=0;j<tupe;j++){
		t_f[i*tupe+j].x=t1[j*bat+i];
		t_f[i*tupe+j].y=0;
		}
	}

	//transform t2
	for(int i=0;i<bat2;i++){
	   for(int j=0;j<tupe;j++){
		t_f2[i*tupe+j].x=t2[j*bat2+i];
		t_f2[i*tupe+j].y=0;
		}
	}
    //set stream for t1 and t2
    cudaStream_t* stream = (cudaStream_t*)malloc(PLAN1D_SIZE*sizeof(cudaStream_t));
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamCreate(&stream[i]);
    }

	//tfft:C2C
	cufftComplex* d_fftData;
	cufftComplex* d_fftData2;
	cudaMalloc((void**)&d_fftData,tupe*bat*sizeof(cufftComplex));	
	cudaMalloc((void**)&d_fftData2,tupe*bat2*sizeof(cufftComplex));
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

    //process bat2
    int bat2_s=bat2%PLAN1D_SIZE;
    int bat2_num=bat2/PLAN1D_SIZE;
	cufftHandle * plan2=(cufftHandle*)malloc(sizeof(cufftHandle)*PLAN1D_SIZE);
    memset(plan2,0,sizeof(cufftHandle));

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftPlan1d(&plan2[i],tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
	 	fprintf(stdout,"[%s]:[%d] cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;
	}
        cufftSetStream(plan2[i],stream[i]);
    }

    if(bat2_num > 0){
    for(int j=0;j<bat2_num;j++){

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(d_fftData2+i*tupe+j*tupe*PLAN1D_SIZE,t_f2+i*tupe+j*tupe*PLAN1D_SIZE,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]);
     }

    #pragma unroll	
	for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(plan2[i],d_fftData2+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData2+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
            	}
        	}
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(t_f2+i*tupe+j*tupe*PLAN1D_SIZE,d_fftData2+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
         }
    }

    #pragma unroll
    for(int i=0;i<bat2_s;i++){
	cudaMemcpyAsync(d_fftData2+i*tupe+bat2_num*tupe*PLAN1D_SIZE,t_f2+i*tupe+bat2_num*tupe*PLAN1D_SIZE,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]);
    }
    #pragma unroll	
	for(int i=0;i<bat2_s;i++){
	if(cufftExecC2C(plan2[i],d_fftData2+i*tupe+bat2_num*tupe*PLAN1D_SIZE,d_fftData2+i*tupe+PLAN1D_SIZE*bat2_num*tupe,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
    #pragma unroll
    for(int i=0;i<bat2_s;i++){
	cudaMemcpyAsync(t_f2+i*tupe+bat2_num*tupe*PLAN1D_SIZE,d_fftData2+i*tupe+tupe*bat2_num*PLAN1D_SIZE,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
    }
    }else{
    #pragma unroll
    for(int i=0;i<bat2_s;i++){
	cudaMemcpyAsync(d_fftData2+i*tupe,t_f2+i*tupe,tupe*sizeof(cufftComplex),cudaMemcpyHostToDevice,stream[i]);
    }
    #pragma unroll	
	for(int i=0;i<bat2_s;i++){
	if(cufftExecC2C(plan2[i],d_fftData2+i*tupe,d_fftData2+i*tupe,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
    #pragma unroll
    for(int i=0;i<bat2_s;i++){
	cudaMemcpyAsync(t_f2+i*tupe,d_fftData2+i*tupe,sizeof(cufftComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
    }
    }

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }

	//transform
    cufftComplex* t_f3 = (cufftComplex*)malloc(sizeof(cufftComplex)*tupe*bat);
	cufftComplex* t_f4 = (cufftComplex*)malloc(sizeof(cufftComplex)*tupe*bat2);

	for(int i=0;i<bat;i++){
	  for(int j=0;j<tupe;j++){
		t_f3[j*bat+i]=t_f[i*tupe+j];
		}
	}
	for(int i=0;i<bat2;i++){
	  for(int j=0;j<tupe;j++){
		t_f4[j*bat2+i]=t_f2[i*tupe+j];
		}
	}
	cudaMemcpy(d_fftData,t_f3,sizeof(cufftComplex)*bat*tupe,cudaMemcpyHostToDevice);
	cudaMemcpy(d_fftData2,t_f4,sizeof(cufftComplex)*bat2*tupe,cudaMemcpyHostToDevice);
    //destroy plan1 and plan2
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftDestroy(plan[i])!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cufftDestroy(plan2[i])!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
    }
	if(t_f != NULL){
	cudaFreeHost(t_f);
	t_f = NULL;
	}
	if(t_f2 !=NULL){
	cudaFreeHost(t_f2);
	t_f2 = NULL;	
	}
	if(t_f3 != NULL){
	free(t_f3);
	t_f3 = NULL;
	}
	if(t_f4 != NULL){
	free(t_f4);
	t_f4 = NULL;
	}
    if(plan != NULL){
    free(plan);
    plan=NULL;
    }
    if(plan2 != NULL){
    free(plan2);
    plan2=NULL;
    }

	//gemmbatched

	cufftComplex* d_Tf;
 	cudaMalloc((void**)&d_Tf,tupe*row*col*sizeof(cufftComplex));
	cublasHandle_t* handle=(cublasHandle_t *)malloc(PLAN1D_SIZE*sizeof(cublasHandle_t));
    memset(handle,0,sizeof(cublasHandle_t));
	cuComplex alpha;
	alpha.x =1;
	alpha.y =0;
	cuComplex beta;
	beta.x = 0;
	beta.y = 0;
	int Am = row;
	int An = rank;
	int Bn = col;
	int Bm = rank;
	int strA = Am*An;
	int strB = Bm*Bn;
	int strC = Am*Bn;
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cublasCreate(&handle[i]);
    cublasSetStream(handle[i],stream[i]);
    }
    int tupe_num=tupe/PLAN1D_SIZE;
    int tupe_s=tupe%PLAN1D_SIZE;
    if(tupe_num > 0){
    #pragma unroll
    for(int j=0;j<tupe_num;j++){
    #pragma unroll
	for(int i=0; i<PLAN1D_SIZE; i++){
	if(cublasCgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
	        &alpha, d_fftData+i*strA+strA*j*PLAN1D_SIZE, Am,d_fftData2+i*strB+strB*j*PLAN1D_SIZE, Bm,  &beta,
	        d_Tf+i*strC+j*strC*PLAN1D_SIZE, Am) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cublasCgemm failed!",__FUNCTION__,__LINE__);
		return;
        	}
	  }
    }
    #pragma unroll
	for(int i=0; i<tupe_s; i++){
	if(cublasCgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
	        &alpha, d_fftData+i*strA+strA*tupe_num*PLAN1D_SIZE, Am,d_fftData2+i*strB+strB*tupe_num*PLAN1D_SIZE, Bm,  &beta,
	        d_Tf+i*strC+tupe_num*strC*PLAN1D_SIZE, Am) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cublasCgemm failed!",__FUNCTION__,__LINE__);
		return;
	}
	  }
    }else{
    #pragma unroll
	for(int i=0; i<tupe_s; i++){
	if(cublasCgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
	        &alpha, d_fftData+i*strA, Am,d_fftData2+i*strB, Bm,  &beta,
	        d_Tf+i*strC, Am) !=CUBLAS_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cublasCgemm failed!",__FUNCTION__,__LINE__);
		return;
	}
	  }
    }
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cublasDestroy(handle[i]);
    }


	cudaFree(d_fftData);
	cudaFree(d_fftData2);

	//Tifft

	cuComplex* host_result=(cuComplex*)malloc(sizeof(cuComplex)*tupe*row*col);
	cuComplex* host_result2;

    cudaHostAlloc((void**)&host_result2,sizeof(cuComplex)*tupe*row*col,cudaHostAllocDefault);

	cudaMemcpy(host_result,d_Tf,sizeof(cuComplex)*tupe*row*col,cudaMemcpyDeviceToHost);

	//transform
	int bat3=row*col;
	for(int i=0;i<bat3;i++){
	  for(int j=0;j<tupe;j++){
		host_result2[i*tupe+j]=host_result[j*bat3+i];
		}
	}
	cufftHandle* iplan = (cufftHandle*)malloc(PLAN1D_SIZE*sizeof(cufftHandle));
    memset(iplan,0,sizeof(cufftHandle));

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftPlan1d(&iplan[i],tupe,CUFFT_C2C,1) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cufftPlan1d failed!",__FUNCTION__,__LINE__);
		return;	
	}
    cufftSetStream(iplan[i],stream[i]);
    }
	
	//ifft
    int bat3_num=bat3/PLAN1D_SIZE;
    int bat3_s=bat3%PLAN1D_SIZE;
    if(bat3_num > 0){
    #pragma unroll
    for(int j=0;j<bat3_num;j++){
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(d_Tf+i*tupe+j*tupe*PLAN1D_SIZE,host_result2+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
	for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftExecC2C(iplan[i],d_Tf+i*tupe+j*tupe*PLAN1D_SIZE,d_Tf+i*tupe+j*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	cudaMemcpyAsync(host_result2+i*tupe+j*tupe*PLAN1D_SIZE,d_Tf+i*tupe+j*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
    }
    }
    #pragma unroll
    for(int i=0;i<bat3_s;i++){
	cudaMemcpyAsync(d_Tf+i*tupe+bat3_num*tupe*PLAN1D_SIZE,host_result2+i*tupe+bat3_num*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
    #pragma unroll
	for(int i=0;i<bat3_s;i++){
	if(cufftExecC2C(iplan[i],d_Tf+i*tupe+bat3_num*tupe*PLAN1D_SIZE,d_Tf+i*tupe+bat3_num*tupe*PLAN1D_SIZE,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
        	}
    }
    #pragma unroll
    for(int i=0;i<bat3_s;i++){
	cudaMemcpyAsync(host_result2+i*tupe+bat3_num*tupe*PLAN1D_SIZE,d_Tf+i*tupe+bat3_num*tupe*PLAN1D_SIZE,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
          }

    }else{

    #pragma unroll
    for(int i=0;i<bat3_s;i++){
	cudaMemcpyAsync(d_Tf+i*tupe,host_result2+i*tupe,sizeof(cuComplex)*tupe,cudaMemcpyHostToDevice,stream[i]);
	}
    #pragma unroll
	for(int i=0;i<bat3_s;i++){
	if(cufftExecC2C(iplan[i],d_Tf+i*tupe,d_Tf+i*tupe,CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR:cufftExecC2C failed!",__FUNCTION__,__LINE__);
		return;
        	}
    }
    #pragma unroll
    for(int i=0;i<bat3_s;i++){
	cudaMemcpyAsync(host_result2+i*tupe,d_Tf+i*tupe,sizeof(cuComplex)*tupe,cudaMemcpyDeviceToHost,stream[i]);
          }
	}

    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
        cudaStreamSynchronize(stream[i]);
    }
    #pragma unroll
    for(int i=0;i<PLAN1D_SIZE;i++){
	if(cufftDestroy(iplan[i])!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cufftDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
        cudaStreamDestroy(stream[i]);
    }
	//transform
	for(int i=0;i<bat3;i++){
	  for(int j=0;j<tupe;j++){
		T[j*bat3+i]=host_result2[i*tupe+j].x/tupe;
		}
	}

    if(stream != NULL){
    free(stream);
    stream=NULL;
    } 
	if(host_result != NULL){
	free(host_result);
	host_result = NULL;
	}

	if(host_result2 != NULL){
	cudaFreeHost(host_result2);
	host_result2 = NULL;
	}

	cudaFree(d_Tf);
}
