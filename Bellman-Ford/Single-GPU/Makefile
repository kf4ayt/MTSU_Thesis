#
#
# You will need to edit the NVCC, INC, LIB, OPT, and ARCH values
# to suit your CUDA install and hardware.
#
#

#####################################
#### Values used for AWS testing ####
#####################################

#NVCC=/usr/local/cuda-11.0/bin/nvcc

#INC=-I/usr/local/cuda-11.0/include
#LIB=-L/usr/local/cuda-11.0/lib64

#OPT=-ccbin /usr/bin/g++

#ARCH=-arch=sm_70

#####################################


NVCC=/usr/bin/nvcc
INC=
LIB=

EXE=SSSP_BF

OPT=-ccbin /usr/bin/g++

ARCH=-arch=sm_61

SUPPORT=BF_file_io.cu BF_print_functions.cu BF_kernels.cu

CUDA_BF: single-gpu_BF_with_APSP.cu BF_file_io.cuh BF_file_io.cu BF_custom_data_structures.cuh BF_print_functions.cuh BF_print_functions.cu BF_kernels.cuh BF_kernels.cu
	${NVCC} ${INC} ${LIB} ${OPT} ${ARCH} -o ${EXE} single-gpu_BF_with_APSP.cu ${SUPPORT} -lcudart


