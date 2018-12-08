#=====================================================
#  CUDA MAKEFILE
#  @AUTHOR: HAILI
#  @DATE: 8.23 2018
#     makefile and src in different dir 
#
#====================================================

include ../../make.inc

src_dir=.
bin_dir=../bin
lib_dir=../lib
inc_dir=../include
src := $(wildcard $(src_dir)/tfft/based/*.cu) \
	$(wildcard $(src_dir)/tfft/streamed/*.cu) \
	$(wildcard $(src_dir)/tfft/batched/*.cu) \
	$(wildcard $(src_dir)/tproduct/based/*.cu) \
	$(wildcard $(src_dir)/tproduct/streamed/*.cu) \
	$(wildcard $(src_dir)/tproduct/batched/*.cu) \
	$(wildcard $(src_dir)/tproduct/*.cu) \
	$(wildcard $(src_dir)/tqr/based/*cu) \
	$(wildcard $(src_dir)/tqr/streamed/*cu) \
	$(wildcard $(src_dir)/tqr/batched/*cu) \
	$(wildcard $(src_dir)/tsvd/based/*cu) \
	$(wildcard $(src_dir)/tsvd/streamed/*cu) \
	$(wildcard $(src_dir)/tsvd/batched/*cu) \
	$(wildcard $(src_dir)/tinv/batched/*cu) \
	$(wildcard $(src_dir)/tinv/based/*cu) \
	$(wildcard $(src_dir)/tinv/streamed/*cu) \
	$(wildcard $(src_dir)/tnorm/batched/*cu) \
	$(wildcard $(src_dir)/based/*.cu)
 
obj  += $(patsubst %.cu,$(bin_dir)/%.o,$(notdir $(src)))

AR = $(ARCH)
#TARGET =$(obj)
TARGET = $(lib_dir)/libcuoperations.a
#CUNV = nvcc
#INCLUDE = -I$(inc_dir) -I/usr/loacl/cuda/include -I/usr/local/magma/include -I/home/haili/Downloads/kblas-gpu-master/include
#LIB = -L /usr/local/magma/lib -L /home/haili/Downloads/kblas-gpu-master/lib
#FLAGS = -DADD_  -std=c++11
#LINK = -lcusolver -lcusparse -lcublas -lcudart -lcufft -lcurand
#LINK += -lmagma  -lkblas-gpu

CUNV    = $(NVCC)
INCLUDE = -I$(inc_dir) $(INC)
LIB     = $(LIBDIR)
FLAGS   = $(NVCCFLAGS) $(CXXFLAGS) 
CODE_FLAGS = $(GENCODE_FLAGS)
LINK    = $(LINKLIB)

ALL:$(TARGET)
$(bin_dir)/%.o:$(src_dir)/tfft/based/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tfft/streamed/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tfft/batched/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tproduct/based/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tproduct/streamed/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tproduct/batched/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tproduct/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tqr/based/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tqr/streamed/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tqr/batched/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tinv/based/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tinv/streamed/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tinv/batched/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tsvd/batched/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tsvd/based/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tsvd/streamed/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/tnorm/batched/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/based/%.cu
	$(CUNV) $(INCLUDE) $(CODE_FLAGS) -c -o $(bin_dir)/$@  $<
$(TARGET):$(obj)
	$(AR) rs $@ $^
	@echo "make file success!"
.PHONY:clean
clean:
	-rm -rf $(obj) $(TARGET)
