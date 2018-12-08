# programs

CC         = gcc
CXX        = g++
NVCC       = nvcc

ARCH       = ar

#--------------------
# flags

NVCCFLAGS  += -O3
CXXFLAGS   :=  -DADD_ -std=c++11 
SM         = 60
GENCODE_FLAGS += -arch 'compute_$(SM)' -code 'compute_$(SM)'
CFLAGS     += -std=c99    

#--------------------
#libraries

LINKLIB    += -lcudart -lcufft -lcublas -lcusparse \
	      -lcusolver \
	      -lcurand -lmagma -lkblas-gpu

#--------------------
#directories

MAGMADIR   ?= /usr/local/magma
CUDADIR    ?= /usr/local/cuda
KBLASDIR   ?= /home/haili/Downloads/kblas-gpu-master

LIBDIR     += -L$(CUDADIR)/lib64 \
	      -L$(MAGMADIR)/lib  \
	      -L$(KBLASDIR)/lib


INC        += -I$(CUDADIR)/include \
	      -I$(MAGMADIR)/include \
	      -I$(KBLASDIR)/include 
