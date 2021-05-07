

######################################
#                                    #
# Values used for AWS                #
#                                    #
# NVCC=/usr/local/cuda-11.0/bin/nvcc #
#                                    #
# INC=-I/usr/local/cuda-11.0/include #
# LIB=-L/usr/local/cuda-11.0/lib64   #
#                                    #
# ARCH=-arch=sm_70                   #
#                                    #
######################################


EXE=blocked_FW

OPT=-ccbin /usr/bin/g++

GPP=/usr/bin/g++
NVCC=/usr/bin/nvcc

INC=
LIB=

ARCH=-arch=sm_61

PROG=objects/kernels.o objects/program.o
SUPPORT=objects/file_io.o


all: kernels.o program.o file_io.o
	${GPP} ${INC} ${LIB} ${PROG} ${SUPPORT} -o ${EXE} -lcudart

kernels.o: kernels.cu
	${NVCC} ${INC} ${LIB} ${OPT} ${ARCH} -c kernels.cu -o objects/kernels.o

program.o: single_gpu_FW_blocked.cpp
	${GPP} ${INC} ${LIB} -c single_gpu_FW_blocked.cpp -o objects/program.o

file_io.o: FW_file_io.cpp
	${GPP} ${INC} ${LIB} -c FW_file_io.cpp -o objects/file_io.o


