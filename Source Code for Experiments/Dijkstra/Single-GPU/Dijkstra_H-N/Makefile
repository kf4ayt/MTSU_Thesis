

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


EXE=SSSP_HN

OPT=-ccbin /usr/bin/g++

GPP=/usr/bin/g++
NVCC=/usr/bin/nvcc

INC=
LIB=

ARCH=-arch=sm_61


PROG=objects/dpKernel.o objects/H-N_kernels.o objects/program.o
SUPPORT=objects/file_io.o objects/print.o
ALG=objects/cpu_dijkstra.o


all: dpKernel.o H-N_kernels.o program.o file_io.o print.o cpu_dijkstra.o
	${GPP} ${INC} ${LIB} ${PROG} ${SUPPORT} ${ALG} -o ${EXE} -lcudart

H-N_kernels.o: H-N_kernels.cu
	${NVCC} ${INC} ${LIB} ${OPT} ${ARCH} -c H-N_kernels.cu -o objects/H-N_kernels.o

dpKernel.o: dpKernel.cu
	${NVCC} ${INC} ${LIB} ${OPT} ${ARCH} -c dpKernel.cu -o objects/dpKernel.o

program.o: HN_single-GPU.cpp
	${GPP} ${INC} ${LIB} -c HN_single-GPU.cpp -o objects/program.o

file_io.o: Dijkstra_file_io.cpp
	${GPP} ${INC} ${LIB} -c Dijkstra_file_io.cpp -o objects/file_io.o

print.o: Dijkstra_print_functions.cpp
	${GPP} ${INC} ${LIB} -c Dijkstra_print_functions.cpp -o objects/print.o

cpu_dijkstra.o: dijkstra-classic-v2.cpp
	${GPP} ${INC} ${LIB} -c dijkstra-classic-v2.cpp -o objects/cpu_dijkstra.o


