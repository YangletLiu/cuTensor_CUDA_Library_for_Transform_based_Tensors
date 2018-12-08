#include "one_step.h"
#include "Tfft.h"
void one_step(cufftComplex* T_omega_f, cufftComplex* omega_f, cufftComplex* X_f, cufftComplex* Y_f, int m, int n, int k,int r_)
{
    cuComplex *tensor_V = new cuComplex[k*m];
    cuComplex *temp = new cuComplex[k*m*r_*k];

    for(int i = 0; i < n; i++){
   
        
          for(int it=0;it<k*m*r_*k;it++){
            temp[it].x=0;
            temp[it].y=0;
        }
//        double tempS = cpuSecond();
        for(int ri = 0; ri < r_; ri++)
            for(int k2 = 0; k2 < k; k2++)
                for(int k1 = 0; k1 < k; k1++)
                    for(int mi =0; mi < m; mi++){
                        int col = r_ * k2 + ri;
                        int row = ((k1 + k2)%k) * m + mi;
                        mul_cufft(omega_f + k1*m*n + i*m + mi, X_f +k2*m*r_ + ri*m + mi,temp + col*m*k + row);
                        // temp[col*m*k + row] = t[k1*m*n + mi] * t[k2*m*n + ri*m + mi];
                    }
        // cout << "temp " <<i << endl;
        // TprintTensor(k*m,k*r_,1,temp);
       /* double tempE = cpuSecond() - tempS;
        printf("Time of temp %f sec\n", tempE);
       */
        for(int j = 0; j < k; j++)
            for(int it = 0; it < m;it++)
                tensor_V[j*m + it]=T_omega_f[j*m*n + i*m + it];
        // cout << "tensor_V" << endl;
        // TprintTensor(k, m, 1,tensor_V); 
  //      double qrS = cpuSecond();  
        cusolverDnHandle_t cusolverH = NULL;
        cublasHandle_t cublasH = NULL;
        cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    
        cudaError_t cudaStat1 = cudaSuccess;
        cudaError_t cudaStat2 = cudaSuccess;
        cudaError_t cudaStat3 = cudaSuccess;
        cudaError_t cudaStat4 = cudaSuccess;

        cuComplex *d_A = NULL; // linear memory of GPU  
        cuComplex *d_tau = NULL; // linear memory of GPU 
        cuComplex *d_B  = NULL; 
        int *devInfo = NULL; // info in gpu (device copy)
        cuComplex *d_work = NULL;
        int  lwork = 0; 

//        int info_gpu = 0;

        cuComplex one;
        one.x = 1;
        one.y =0;

        // step 1: create cusolver/cublas handle
        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        // step 2: copy A and B to device
        cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(cuComplex) * m*k*r_*k);
        cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(cuComplex) * r_*k);
        cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(cuComplex) * m*k);
        cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);
        assert(cudaSuccess == cudaStat4);

        cudaStat1 = cudaMemcpy(d_A, temp, sizeof(cuComplex) *m*k*r_*k, cudaMemcpyHostToDevice);
        cudaStat2 = cudaMemcpy(d_B, tensor_V, sizeof(cuComplex) * m*k, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);

        // step 3: query working space of geqrf and ormqr
        cusolver_status = cusolverDnCgeqrf_bufferSize(
            cusolverH, 
            m*k, 
            k*r_, 
            d_A, 
            m*k, 
            &lwork);
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuComplex)*lwork);
        assert(cudaSuccess == cudaStat1);

        // step 4: compute QR factorization
        cusolver_status = cusolverDnCgeqrf(
            cusolverH, 
            m*k, 
            k*r_, 
            d_A, 
            m*k, 
            d_tau, 
            d_work, 
            lwork, 
            devInfo);
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        assert(cudaSuccess == cudaStat1);

        // check if QR is good or not
        // cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        // assert(cudaSuccess == cudaStat1);

        // printf("after geqrf: info_gpu = %d\n", info_gpu);
        // assert(0 == info_gpu);

        // step 5: compute Q^T*B
        cusolver_status= cusolverDnCunmqr(
            cusolverH, 
            CUBLAS_SIDE_LEFT, 
            CUBLAS_OP_C,
            m*k, 
            1, 
            k*r_, 
            d_A, 
            m*k,
            d_tau,
            d_B,
            m*k,
            d_work,
            lwork,
            devInfo);
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        assert(cudaSuccess == cudaStat1);
        // check if QR is good or not
        // cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        // assert(cudaSuccess == cudaStat1);

        // printf("after ormqr: info_gpu = %d\n", info_gpu);
        // assert(0 == info_gpu);
// step 6: compute x = R \ Q^T*B

        cublas_status = cublasCtrsm(
            cublasH,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, 
            CUBLAS_DIAG_NON_UNIT,
            k*r_,
            1,
            &one,
            d_A,
            m*k,
            d_B,
            m*k);
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        assert(cudaSuccess == cudaStat1);
        cuComplex temp_Y_f[r_*k];
        cudaStat1 = cudaMemcpy(temp_Y_f, d_B, sizeof(cuComplex)*r_*k, cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat1);
        cudaMemcpy(temp, d_A, sizeof(cuComplex)*m*k, cudaMemcpyDeviceToHost);

        if (d_A    ) cudaFree(d_A);
        if (d_tau  ) cudaFree(d_tau);
        if (d_B    ) cudaFree(d_B);
        if (devInfo) cudaFree(devInfo);
        if (d_work ) cudaFree(d_work);


        if (cublasH ) cublasDestroy(cublasH);   
        if (cusolverH) cusolverDnDestroy(cusolverH);   
        // cudaDeviceReset();
    //     Y_f(:,i,j) = temp_Y_f((j-1)*r + 1 : j*r);
    // end
        // cout << "temp_Y_f" << endl;
        // TprintTensor(r_, k, 1,temp_Y_f);
        // cout << "temp" << endl;
        // TprintTensor(k*m,k*r_,1,temp);
/*        double qrE = cpuSecond() - qrS;
        printf("Time of QR %f sec\n", qrE);
  */
        for (int j = 0; j < k; j++)
        	for (int a =0; a < r_; a++)
        		Y_f[j*r_*n + i*r_ + a] = temp_Y_f[j*r_ + a];
    }
    delete[] temp;
    temp = NULL;
    delete[] tensor_V;
    tensor_V = NULL;
}
