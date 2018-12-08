#=====================================================
#  CUDA NAKEFILE
#  @AUTHOR: HAILI
#  @DATE: 8.26 2018
#     makefile and src in different dir 
#
#====================================================
src_path=..
obj_path=.
src = $(wildcard $(src_path)/*.cpp)
src1 = $(wildcard $(src_path)/*.cu)
obj = $(patsubst %.cpp,$(obj_path)/%.o,$(notdir $(src)))
obj1 = $(patsubst %.cu,$(obj_path)/%.o,$(notdir $(src1)))
CC = gcc
CUNV = nvcc
INCLUDE = -I ../../../include
INCLUDE += -I ../
LIB= -L ../../../lib -lcuoperations -lcublas -lcusparse -lcudart -lcufft -lcusolver
LINK = -link
target = test
ALL:$(target)
%.o:$(src_path)/%.cu
	$(CUNV) $< $(INCLUDE) $(LIB) -c $@
%.o:$(src_path)/%.cpp
	$(CUNV) $< $(INCLUDE) $(LIB) -c $@
$(target):$(obj) $(obj1)
	$(CUNV)  $(LIB) $(LINK) -o $@ $(obj) $(obj1)
	@echo 'make file success!'
.PHONY:clean
clean:
	-rm -rf test  $(obj) $(obj1)

