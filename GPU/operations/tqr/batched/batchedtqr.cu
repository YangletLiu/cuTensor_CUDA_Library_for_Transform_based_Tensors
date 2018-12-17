#include"batchedtqr.h"
void batchedtqr(float* t,const int m,const int n,const int tupe,cuComplex* tau)
{
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
	if(cufftGetSizeMany(plan,1,n_f,in,stride,dist,on,stride,dist,
			CUFFT_R2C,bat,&worksize)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Estimate work size failed!",__FUNCTION__,__LINE__);
		return;
 	}
//	printf("the work size is:%lf G\n",(double)worksize/(1024*1024*1024));

	if(cufftExecR2C(plan,d_t,(cufftComplex*)d_fftData)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	if(d_t !=NULL){
	cudaFree(d_t);
    d_t=NULL;   
    }
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]cufftDestory faile!",__FUNCTION__,__LINE__);
		return;
	}

    if(magma_init() != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_init error!",__FUNCTION__,__LINE__);
		return;
    }
    magma_queue_t queue=NULL;
    magma_int_t dev = 0;
    magma_queue_create(dev, &queue);
    
//	    magmaFloatComplex *h_Amagma;
// 	    magmaFloatComplex *htau_magma;
    magmaFloatComplex *d_A, *dtau_magma;
    magmaFloatComplex **dA_array = NULL;
    magmaFloatComplex **dtau_array = NULL;

    magma_int_t   *dinfo_magma;
    magma_int_t M, N, lda, ldda, min_mn;
    magma_int_t batchCount;
    magma_int_t column;

    M = m;
    N = n;
    batchCount = ht;
    min_mn = min(M, N);
    lda    = M;
//            n2     = lda * N * batchCount;
//    ldda = ((M+31)/32)*32;
    ldda = magma_roundup( M, 32 );
//            magma_cmalloc_cpu( &h_Amagma,   n2     );
//            magma_cmalloc_cpu( &htau_magma, min_mn * batchCount );
     if(magma_cmalloc( &d_A,   ldda*N * batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }

     if(magma_cmalloc( &dtau_magma,  min_mn * batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }

     if(magma_imalloc( &dinfo_magma,  batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
 
     if(magma_malloc((void**) &dA_array,   batchCount * sizeof(magmaFloatComplex*) ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
     if(magma_malloc((void**) &dtau_array, batchCount * sizeof(magmaFloatComplex*) ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
     column = N * batchCount;

     magma_ccopymatrix(M, column, d_fftData, M, d_A, ldda, queue );
	
//   magma_cprint_gpu(M, column, d_fftData, M, queue );
//   magma_cprint_gpu(M, column, d_A, ldda, queue );
         
     magma_cset_pointer( dA_array, d_A, 1, 0, 0, ldda*N, batchCount, queue );
     magma_cset_pointer( dtau_array, dtau_magma, 1, 0, 0, min_mn, batchCount, queue );
  
//   magma_cprint_gpu(M, column, d_A, ldda, queue );

//   magma_cprint_gpu(M, column, d_fftData, M, queue );

    if( magma_cgeqrf_batched(M, N, dA_array, ldda, dtau_array, dinfo_magma, batchCount, queue) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_cgeqrf_batched!",__FUNCTION__,__LINE__);
		return;
    }
//   magma_cprint_gpu(M, column, d_A, ldda, queue );
//         magma_cgetmatrix( M, column, d_A, ldda, h_Amagma, lda, queue );
//   magma_cgetmatrix(min_mn, batchCount, dtau_magma, min_mn, htau_magma, min_mn, queue );
     magma_cgetmatrix(min_mn, batchCount, dtau_magma, min_mn, tau, min_mn, queue );
     
//   magma_cprint( M, column, h_Amagma, lda);
//   magma_cprint(min_mn, batchCount, htau_magma, min_mn);

     magma_ccopymatrix(M, column, d_A, ldda, d_fftData, lda, queue );
     magma_queue_destroy( queue );
     if( d_A != NULL ){ 
     magma_free( d_A   );
     d_A = NULL;
     }
     if( dtau_magma != NULL ){
     magma_free( dtau_magma  );
     dtau_magma = NULL;
     }
     if( dinfo_magma != NULL){
     magma_free( dinfo_magma );
     dinfo_magma = NULL;
     }
     if( dA_array != NULL){
     magma_free( dA_array   );
     dA_array = NULL;
     }
     if( dtau_array != NULL){
     magma_free( dtau_array  );
     dtau_array = NULL;
     }
     if( magma_finalize() != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_finalize error!",__FUNCTION__,__LINE__);
		return;
     }
	
	cufftHandle iplan =0;
	in[0] = ht;
	on[0] = tupe;
	
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
	
	float* d_qr;
	cudaMalloc((void**)&d_qr,sizeof(float)*tupe*bat);
	if(cufftExecC2R(iplan,(cufftComplex*)d_fftData,d_qr)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

    int num=bat*tupe;
	int threads,blocks;
    if(num<512){
        threads=num;
        blocks=1;
    }else{
	    threads=512;
	    blocks=((num%512 ==0)?num/512:num/512+1);
	}
        fftResultProcess<<<blocks,threads>>>(d_qr,num,tupe);

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaMemcpy(t,d_qr,sizeof(float)*bat*tupe,cudaMemcpyDeviceToHost);
    if(d_qr !=NULL){
    cudaFree(d_qr);
    d_qr=NULL;
    }
    if(d_fftData!=NULL){
	cudaFree(d_fftData);
    d_fftData=NULL;
    }
cudaDeviceReset();            
}
