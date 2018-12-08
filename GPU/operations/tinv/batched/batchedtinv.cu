#include"inv.h" 
void batchedtinv(float* t,const int m,const int n,const int tupe,float* invA){
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
	
	cudaFree(d_t);
	if(cufftDestroy(plan)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d]cufftDestory faile!",__FUNCTION__,__LINE__);
		return;
	}
	
	
	magma_init();
	magma_queue_t queue = NULL;
	magma_int_t dev = 0;
	magma_queue_create( dev, &queue);


	magmaFloatComplex *d_A, *d_invA;
	magmaFloatComplex_ptr *dA_array;
	magmaFloatComplex_ptr *dinvA_array;
	magma_int_t **dipiv_array;
	magma_int_t *dinfo_array;
	magma_int_t *d_ipiv, *d_info;
	magma_int_t M, N, lda, ldda;
	magma_int_t columns;

	M = m;
	N = n;
	magma_int_t batchCount = ht;
	lda = M;
	ldda = magma_roundup( M, 32 );

	magma_cmalloc( &d_A,      ldda*N * batchCount );
	magma_cmalloc( &d_invA,   ldda*N * batchCount );
	magma_imalloc( &d_ipiv,   N * batchCount );
	magma_imalloc( &d_info,   batchCount );

	magma_malloc( (void**) &dA_array,   batchCount * sizeof(magmaFloatComplex*));
	magma_malloc( (void**) &dinvA_array,batchCount * sizeof(magmaFloatComplex*));
	magma_malloc( (void**) &dipiv_array,batchCount * sizeof(magma_int_t*));
	magma_imalloc( &dinfo_array, batchCount);
	
	columns = N * batchCount;
	
	magma_ccopymatrix( M, columns, d_fftData, lda, d_A, ldda, queue );
	
	magma_cset_pointer( dA_array, d_A, ldda, 0, 0, ldda * N, batchCount, queue );
	magma_cset_pointer( dinvA_array, d_invA, ldda, 0, 0, ldda * N, batchCount, queue );
	magma_iset_pointer( dipiv_array, d_ipiv, 1, 0, 0, N, batchCount, queue );

//	magma_cprint_gpu( M, columns, d_A, ldda, queue );

	
	magma_cgetrf_batched( M, N, dA_array, ldda, dipiv_array, dinfo_array, batchCount, queue );
	magma_cgetri_outofplace_batched( M, dA_array, ldda, dipiv_array, dinvA_array, ldda, dinfo_array, batchCount, queue );
	
//	magma_cprint_gpu( M, columns, d_invA, ldda, queue );
	
	magma_ccopymatrix( M, columns, d_invA, ldda, d_fftData, lda, queue );
	
	magma_queue_destroy( queue );
	magma_free( d_A   );
        magma_free( d_invA );
        magma_free( dA_array   );
        magma_free( dinvA_array  );
	magma_free( d_ipiv );
	magma_free( d_info );
	magma_free( dipiv_array );
	magma_free( dinfo_array );
        magma_finalize();
	
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
//	printf("the work size is:%ld G\n",(double)worksize/(1024*1024*1024));
	
	float* d_inv;
	cudaMalloc((void**)&d_inv,sizeof(float)*tupe*bat);
	if(cufftExecC2R(iplan,(cufftComplex*)d_fftData,d_inv)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
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
         fftResultProcess<<<blocks,threads>>>(d_inv,num,tupe);

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaMemcpy(invA,d_inv,sizeof(float)*bat*tupe,cudaMemcpyDeviceToHost);
        
	cudaFree(d_inv);
	cudaFree(d_fftData); 
        
	}
