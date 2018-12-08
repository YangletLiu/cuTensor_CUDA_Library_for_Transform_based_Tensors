include ../../../../make.inc

src_path = ..
obj_path = .

src = $(wildcard $(src_path)/*.cpp)
obj = $(patsubst %.cpp,%.o,$(notdir $(src)))

CC = gcc
CUNV = nvcc
INCLUDE = -I ../../../include $(INC)
LIB  = -L ../../../lib $(LIBDIR) $(LINKLIB) -lcuoperations \
	-lcusolver -lcusparse -lcublas -lcufft -lcudart
LINK = -link
FLAGS = -DADD -std=c++11
target = result

all:$(target)
%.o:$(src_path)/%.cpp
	$(CUNV) $< -c $(LIB) $(INCLUDE) $(FLAGS) $@
$(target):$(obj)
	$(CUNV) $(LIB) $(INCLUDE) $(FLAGS) $(LINK) -o $@ $^

.PHONY:clean
clean:
	-rm -rf result $(obj)
