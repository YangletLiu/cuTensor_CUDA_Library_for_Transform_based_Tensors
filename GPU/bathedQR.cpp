#include <stdio.h>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_lapack.h"
#define min(a, b) (((a)<(b))?(a):(b))
#define NUM 512

int main(int argc, char **argv)
{
    for (int i=0; i<20; i++)
    {
            int l = (i+1)*50;
            magma_init();
            magma_queue_t queue=NULL;
            magma_int_t dev = 0;
            magma_queue_create(dev, &queue);
            double magma_time;
            magmaFloatComplex *h_A, *h_R, *h_Amagma;
            magmaFloatComplex *d_A, *dtau_magma;
            magmaFloatComplex **dA_array = NULL;
            magmaFloatComplex **dtau_array = NULL;

            magma_int_t   *dinfo_magma;
            magma_int_t M, N, lda, ldda, lwork, n2, info, min_mn;
            magma_int_t ione     = 1;
            magma_int_t ISEED[4] = {0,0,0,1};
            magma_int_t batchCount;
            magma_int_t column;

            M = l;
            N = l;
            batchCount = NUM;
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N * batchCount;
            ldda = ((M+31)/32)*32;
            magma_cmalloc_cpu( &h_A,   n2     );
            magma_cmalloc_cpu( &h_Amagma,   n2     );
            magma_cmalloc_pinned( &h_R,   n2     );
            magma_cmalloc( &d_A,   ldda*N * batchCount );

            magma_cmalloc( &dtau_magma,  min_mn * batchCount );

            magma_imalloc( &dinfo_magma,  batchCount );

            magma_malloc((void**) &dA_array,   batchCount * sizeof(magmaFloatComplex*) );
            magma_malloc((void**) &dtau_array, batchCount * sizeof(magmaFloatComplex*) );
            column = N * batchCount;
            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            lapackf77_clacpy( MagmaFullStr, &M, &column, h_A, &lda, h_R, &lda );

            /* ====================================================================
                Performs operation using MAGMA
                =================================================================== */
            magma_csetmatrix( M, column, h_R, lda,  d_A, ldda, queue );
            magma_cset_pointer( dA_array, d_A, 1, 0, 0, ldda*N, batchCount, queue );
            magma_cset_pointer( dtau_array, dtau_magma, 1, 0, 0, min_mn, batchCount, queue );

            magma_time = magma_sync_wtime( queue );

            info = magma_cgeqrf_batched(M, N, dA_array, ldda, dtau_array, dinfo_magma, batchCount, queue);

            magma_time = magma_sync_wtime( queue ) - magma_time;
            printf ( " %dth turn MAGMA time : %7.3f sec .\n" , i, magma_time );

            magma_cgetmatrix( M, column, d_A, ldda, h_Amagma, lda, queue );

            magma_free_cpu( h_A    );
            magma_free_cpu( h_Amagma );
            magma_free_pinned( h_R    );
                
            magma_free( d_A   );
            magma_free( dtau_magma  );

            magma_free( dinfo_magma );

            magma_free( dA_array   );
            magma_free( dtau_array  );

            magma_finalize();
    }
    return EXIT_SUCCESS;
}
