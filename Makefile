ALL:
	make -C ./GPU/operations
	make -C ./GPU/test/tinv/debug
	make -C ./GPU/test/gemm/debug
	make -C ./GPU/test/fftResultProcess/debug
	make -C ./GPU/test/mprod/debug
	make -C ./GPU/test/tfft/debug
	make -C ./GPU/test/tprod/debug
	make -C ./GPU/test/tqr/debug
	make -C ./GPU/test/tsvd/debug
	make -C ./GPU/test/warmup/debug
	make -C ./GPU/test/tnorm/debug
.PHONY:clean

clean:
	make clean -C ./GPU/operations
	make clean -C ./GPU/test/tinv/debug
	make clean -C ./GPU/test/gemm/debug
	make clean -C ./GPU/test/fftResultProcess/debug
	make clean -C ./GPU/test/mprod/debug
	make clean -C ./GPU/test/tfft/debug
	make clean -C ./GPU/test/tprod/debug
	make clean -C ./GPU/test/tqr/debug
	make clean -C ./GPU/test/tsvd/debug
	make clean -C ./GPU/test/warmup/debug
	make clean -C ./GPU/test/tnorm/debug
