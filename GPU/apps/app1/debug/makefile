#=====================================================
#  CUDA NAKEFILE
#  @AUTHOR: HAILI
#  @DATE: 7.10 2018
#     makefile and src in different dir 
#
#====================================================
include ../../../../make.inc

src_path=..
obj_path=.
src = $(wildcard $(src_path)/*.cu)
obj = $(patsubst %.cu,$(obj_path)/%.o,$(notdir $(src)))
src1 = $(wildcard $(src_path)/*.cpp)
obj1 = $(patsubst %.cpp,$(obj_path)/%.o,$(notdir $(src1)))

CC = gcc
CUNV = nvcc
INCLUDE = -I ../../../include $(INC)
LIB= -L ../../../lib $(LIBDIR) $(LINKLIB) -lcuoperations \
	 -lcusolver -lcusparse -lcublas -lcudart -lcufft
LINK = -link
FLAGS = -DADD_ -std=c++11
target = result
ALL:$(target)
%.o:$(src_path)/%.cu
	$(CUNV) $< -c $(LIB) $(INCLUDE) $(FLAGS) $@
%.o:$(src_path)/%.cpp
	$(CUNV) $< -c $(LIB) $(INCLUDE) $(FLAGS) $@
$(target):$(obj) $(obj1)
	$(CUNV)  $(LIB) $(INCLUDE) $(FLAGS) $(LINK) -o $@ $(obj) $(obj1)
	@echo 'make file success!'
.PHONY:clean
clean:
	-rm -rf result $(obj) $(obj1)

