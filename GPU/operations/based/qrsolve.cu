#include"based.h"
__global__ void CopyUpperSubmatrix( const cuComplex* d_in,cuComplex* d_out,
		const int M, const int N, const int subM){
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	if( i < subM*N)
	d_out[(i/subM)*subM+i%subM] = d_in[(i/subM)*M+i%subM];
}

void qrsolve(cuComplex* d_A,cuComplex* d_B,const int m,const int n,const int k,cuComplex* d_X){

const int M = m;
const int N = n;
const int K = k;
const int min = Min(m,n);

//define handles
cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;

//create handles
if( cusolverDnCreate( &cusolverH ) != CUSOLVER_STATUS_SUCCESS ){
	fprintf(stdout,"[%s]:[%d] cusolverDnCreate error!",__FUNCTION__,__LINE__);
	return;
}
if( cublasCreate( &cublasH ) != CUBLAS_STATUS_SUCCESS ){
	fprintf(stderr,"[%s]:[%d] cublasCreate error!",__FUNCTION__,__LINE__);
	return;
}
cuComplex  *d_work, *d_work2, *d_tau;
int *d_devInfo, devInfo;
cudaMalloc( (void**)&d_tau,sizeof(cuComplex)* min);
cudaMalloc( (void**)&d_devInfo, sizeof(int));
int bufsize,bufsize2;

// in-place A=QR
if( cusolverDnCgeqrf_bufferSize(
		cusolverH,
		M,
		N,
		d_A,
		M,
		&bufsize
		) != CUSOLVER_STATUS_SUCCESS ){
	fprintf(stdout,"[%s]:[%d] cusolverDnCgeqrf_bufferSize error!",__FUNCTION__,__LINE__);
	return;
}

cudaMalloc( (void**)&d_work, sizeof(cuComplex)* bufsize);

if( cusolverDnCgeqrf(
		cusolverH,
		M,
		N,
		d_A,
		M,
		d_tau,
		d_work,
		bufsize,
		d_devInfo
		) != CUSOLVER_STATUS_SUCCESS ){
	fprintf(stdout,"[%s]:[%d] cusolverDnCgeqrf error!",__FUNCTION__,__LINE__);
	cudaMemcpy(&devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Info:%d\n",devInfo);
	return;
}
cudaDeviceSynchronize();

if( d_work ) cudaFree(d_work);
//Q`*B
if( cusolverDnCunmqr_bufferSize(
		cusolverH,
		CUBLAS_SIDE_LEFT,
		CUBLAS_OP_C,
		M,
		K,
		min,
		d_A,
		M,
		d_tau,
		d_B,
		M,
		&bufsize2
	) != CUSOLVER_STATUS_SUCCESS){
	fprintf(stdout,"[%s]:[%d] cusolverDnCunmqr_buffersize error!",__FUNCTION__,__LINE__);
	return;
}

if (cudaMalloc((void**)&d_work2, sizeof(cuComplex)* bufsize2) != cudaSuccess){
	fprintf(stdout,"[%s]:[%d] cuda runtime API error!",__FUNCTION__,__LINE__);
	return;
}

if( cusolverDnCunmqr(
		cusolverH,
		CUBLAS_SIDE_LEFT,
		CUBLAS_OP_C,
		M,
		K,
		min,
		d_A,
		M,
		d_tau,
		d_B,
		M,
		d_work2,
		bufsize2,
		d_devInfo)
	!= CUSOLVER_STATUS_SUCCESS){
	fprintf(stdout,"[%s]:[%d] cusolverDnCunmqr error!",__FUNCTION__,__LINE__);
	cudaMemcpy(&devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Info:%d\n",devInfo);
	return;
}
cudaDeviceSynchronize();

if( d_work2 ) cudaFree(d_work2);
if(d_tau) cudaFree(d_tau);

if(d_devInfo) cudaFree(d_devInfo);

cuComplex *d_R;
cudaMalloc((void**)&d_R, sizeof(cuComplex)* min * N);
int threads=0;
int blocks=0;
int num = min*N;
if(num<512){
	threads=num;
	blocks=1;
}else{
	threads=512;
	blocks = ((num%512) == 0)?num/512:num/512+1;
}
CopyUpperSubmatrix<<<blocks,threads>>>(d_A, d_R, M, N, min);

cudaDeviceSynchronize();
num = min*K;
if(num<512){
	threads=num;
	blocks=1;
}else{
	threads=512;
	blocks = ((num%512) == 0)?num/512:num/512+1;
}
CopyUpperSubmatrix<<<blocks,threads>>>(d_B, d_X, M, K, min);
cudaDeviceSynchronize();

//solve x = R \ (Q`*B)
cuComplex alphat;
alphat.x = 1;
alphat.y = 0;

if( cublasCtrsm(
		cublasH,
		CUBLAS_SIDE_LEFT,
		CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_N,
		CUBLAS_DIAG_NON_UNIT,
		min,
		K,
		&alphat,
		d_R,
		N,
		d_X,
		N
	) != CUBLAS_STATUS_SUCCESS){
	fprintf(stdout,"[%s]:[%d] cusolverDnCunmqr_buffersize error!",__FUNCTION__,__LINE__);
	return;
}

cudaDeviceSynchronize();

if(d_R) cudaFree(d_R);

//Destroy handles
if( cusolverDnDestroy( cusolverH) != CUSOLVER_STATUS_SUCCESS ){
	fprintf(stdout,"[%s]:[%d] cusolverDnDestroy error!",__FUNCTION__,__LINE__);
	return;
}
if( cublasDestroy( cublasH ) != CUBLAS_STATUS_SUCCESS ){
	fprintf(stderr,"[%s]:[%d] cublasDestroy error!",__FUNCTION__,__LINE__);
	return;
}
}

