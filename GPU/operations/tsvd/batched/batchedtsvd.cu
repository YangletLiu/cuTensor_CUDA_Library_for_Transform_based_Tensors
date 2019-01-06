#include "svd.h"
#include "based.h"
#define KBLAS_Success 1

void batchedtsvd(float* t,const int m,const int n, const int tupe, cuComplex* host_u,float* S){
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
	//construct matrix K

//	cuComplex* h_fft=(cuComplex*)malloc(sizeof(cuComplex)*m*n*ht);
//	cudaMemcpy(h_fft, d_fftData, sizeof(cuComplex)*m*n*ht,cudaMemcpyDeviceToHost);
//	for(int i=0;i<m*n*ht;i++){
//	printf("h_fft %f	%f \n",h_fft[i].x,h_fft[i].y);
//	}

	float* d_k;
	cudaMalloc((void**)&d_k,sizeof(float)*m*n*ht*4);
	
	int threads;
	int blocks;
	int num= (m*n*ht*4);
	if(num < 512){
	 threads = num;
	 blocks = 1;
	}else{
	 threads = 512;
	 blocks = ((num%512 ==0)?num/512:num/512+1);
	}

	conMatrixK<<<blocks,threads>>>(d_fftData,d_k,m,n,ht);
	
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	
	cudaFree(d_fftData);	

//	float* h_k=(float*)malloc(sizeof(float)*4*m*n*ht);
//	cudaMemcpy(h_k, d_k, sizeof(float)*4*m*n*ht,cudaMemcpyDeviceToHost);
//	for(int i=0;i<4*m*n*ht;i++){
//	printf("h_k %f \n",h_k[i]);
//	}
			
	//tsvd
	int M=2*m;
	int N=2*n; 
	
	kblasHandle_t handle;
	kblasCreate( &handle );
	kblasSgesvj_batch_wsquery(handle, M, N, ht);

	if(kblasAllocateWorkspace(handle) != 1){
		fprintf(stdout,"[%s]:[%d] kblas  wsquery err!",__FUNCTION__,__LINE__);
		return;
	}

	int stride_s=((M<N)?M:N);
	float* d_s;
	cudaMalloc((void**)&d_s, sizeof(float)*ht*stride_s);
	if(kblasSgesvj_batch_strided(handle, M, N, d_k, M, M*N, d_s, stride_s, ht) != 1){
		fprintf(stdout,"[%s]:[%d] kblas  svd  err!",__FUNCTION__,__LINE__);
		return;
	}

	
	kblasFreeWorkspace(handle);
	
//	cudaMemcpy(h_k, d_k, sizeof(float)*4*m*n*ht,cudaMemcpyDeviceToHost);
//	for(int i=0;i<4*m*n*ht;i++){
//	printf("take_h_k %f \n",h_k[i]);
//	}
	
	//extract elements 

	cuComplex* d_hu,*d_u;
	cudaMalloc((void**)&d_hu,sizeof(cuComplex)*m*n*ht);
	cudaMalloc((void**)&d_u,sizeof(cuComplex)*m*n*tupe);
	num = 4*m*n*ht;
	if(num < 512){
	 threads = num;
	 blocks = 1;
	}else{
	 threads = 512;
	 blocks = ((num%512 ==0)?num/512:num/512+1);
	}
	extractEvenNumU<<<blocks,threads>>>(d_k,d_hu,m,n,ht);	
	
//	printf("\n++++++++++++++++++++++++++\n");
//	cuComplex* h_u = (cuComplex*)malloc(sizeof(cuComplex)*m*n*ht);
//	cudaMemcpy(h_u, d_hu, sizeof(cuComplex)*m*n*ht,cudaMemcpyDeviceToHost);
//	for(int i=0;i<m*n*ht;i++){
//	printf("take_h_u %f	%f \n",h_u[i].x,h_u[i].y);
//	}
//	printf("\n++++++++++++++++++++++++++\n");

	//symmtricRecoverU

	symmetricRecoverU(d_hu,m,n,tupe,d_u);
	cudaDeviceSynchronize();
//	printf("\n++++++++++++++++++++++++++\n");
//	cuComplex* hh_u = (cuComplex*)malloc(sizeof(cuComplex)*m*n*tupe);
//	cudaMemcpy(hh_u, d_u, sizeof(cuComplex)*m*n*tupe,cudaMemcpyDeviceToHost);
//	for(int i=0;i<m*n*tupe;i++){
//	printf("_sstake_h_u %f	%f \n",hh_u[i].x,hh_u[i].y);
//	}
//	printf("\n++++++++++++++++++++++++++\n");

	cudaFree(d_hu);
	cudaFree(d_k);
	//extract s
	float* ds_extract;
	cudaMalloc((void**)&ds_extract,sizeof(float)*ht*((m<n)?m:n));	

	num = ht*((m<n)?m:n);
	if(num < 512){
	 threads = num;
	 blocks = 1;
	}else{
	 threads = 512;
	 blocks = ((num%512 ==0)?num/512:num/512+1);
	}
	extractEvenNumS<<<blocks,threads>>>(d_s,ds_extract,m,n,ht);
	
//	float* h_s2=(float*)malloc(sizeof(float)*num);
//	cudaMemcpy(h_s2, ds_extract, sizeof(float)*num,cudaMemcpyDeviceToHost);
//	for(int i=0;i<num;i++){
//	printf("take_h_s2 %f \n",h_s2[i]);
//	}

	cudaFree(d_s);	
	//itfft_u

	//set stream
	cudaStream_t stream[2];
	
	#pragma unroll
	for(int i=0;i<2;i++){
		cudaStreamCreate(&stream[i]);
	}

	cuComplex* du;
	cudaMalloc((void**)&du,sizeof(cuComplex)*m*n*tupe);

	cufftHandle iplan =0;
	in[0] = tupe;
	on[0] = tupe;
	int stride_in = 1;
	int dist_in = tupe; 
	bat = m*n;
	stride = bat;
	
	cufftSetStream(iplan,stream[0]);
	if (cufftPlanMany(&iplan,1,n_f,in,stride_in,dist_in,on,stride,dist,
				CUFFT_C2C,bat)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}
//	if(cudaDeviceSynchronize() != cudaSuccess){
//		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
//		return;
//	}
	
	//estimate of the work size
	if(cufftGetSizeMany(iplan,1,n_f,in,stride_in,dist_in,on,stride,dist,
			CUFFT_C2C,bat,&worksize)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Estimate work size failed!",__FUNCTION__,__LINE__);
		return;
 	}
	//printf("the work size is:%ld G\n",(double)worksize/(1024*1024*1024));

	if(cufftExecC2C(iplan,(cufftComplex*)d_u,du,CUFFT_INVERSE)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

//cuComplex* hhu = (cuComplex*)malloc(sizeof(cuComplex)*bat*tupe);
//cudaMemcpy(hhu,du,sizeof(cuComplex)*bat*tupe,cudaMemcpyDeviceToHost);
//printf("hhu_______________________________/n");
//for(int i=0;i<bat*tupe;i++){
//printf("[%f %f]	",hhu[i].x,hhu[i].y);
//}
//printf("hhu_______________________________/n");

	num=bat*tupe;
        if(num<512){
          threads=num;
          blocks=1;
        }else{
	  threads=512;
	  blocks=((num%512 ==0)?num/512:num/512+1);
	}
         fftResultProcess<<<blocks,threads,0,stream[0]>>>(du,num,tupe);

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	cudaMemcpyAsync(host_u,du,sizeof(cuComplex)*m*n*tupe,cudaMemcpyDeviceToHost,stream[0]);	
//	cudaMemcpy(t,du,sizeof(float)*m*n*tupe,cudaMemcpyDeviceToHost);
	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);
	cudaFree(du);
	cudaFree(d_u);


	//itfft_s

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

    	float2cuComplex<<<blocks,threads,0,stream[1]>>>(ds_extract,ht*bat,d_s2);
	
//	if(cudaDeviceSynchronize() != cudaSuccess){
//		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
//		return;
//	}
	cudaFree(ds_extract);

	cufftHandle iplan1 =0;
	cufftSetStream(iplan1,stream[1]);
	
	if (cufftPlanMany(&iplan1,1,n_f,in,stride,dist,on,stride,dist,
				CUFFT_C2R,bat)!=CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Plan creation failed!",__FUNCTION__,__LINE__);
		return;
	}

//	if(cudaDeviceSynchronize() != cudaSuccess){
//		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
//		return;
//	}

	if(cufftExecC2R(iplan1,(cufftComplex*)d_s2,d_s3) != CUFFT_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: Exec failed!",__FUNCTION__,__LINE__);
		return;
	}

//	if(cudaDeviceSynchronize() != cudaSuccess){
//		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
//		return;
//	}
	
       num=bat*tupe;
       if(num<512){
          threads=num;
          blocks=1;
       }else{
	  threads=512;
	  blocks=((num%512 ==0)?num/512:num/512+1);
	}
         fftResultProcess<<<blocks,threads,0,stream[1]>>>(d_s3,bat*tupe,tupe);

//	cudaMemcpy(S,d_s3,sizeof(float)*tupe*bat,cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(S,d_s3,sizeof(float)*tupe*bat,cudaMemcpyDeviceToHost,stream[1]);

	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);

	if(cufftDestroy(iplan)!=CUFFT_SUCCESS){

		fprintf(stdout,"[%s]:[%d]cufftDestory failed!",__FUNCTION__,__LINE__);
		return;
	}
	if(cufftDestroy(iplan1)!=CUFFT_SUCCESS){

		fprintf(stdout,"[%s]:[%d]cufftDestory failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	cudaFree(d_s3);
	cudaFree(d_s2);
	
	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	
	cudaDeviceSynchronize();
}
