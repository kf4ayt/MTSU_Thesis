

##########################################
#                                        #
# Values used for AWS                    #
#                                        #
# COMPILER=/usr/local/cuda-11.0/bin/nvcc #
#                                        #
# ARCH=-arch=sm_70                       #
#                                        #
##########################################


COMPILER=/usr/bin/nvcc

EXE=CUDA_FW

OPT=-ccbin /usr/bin/g++

ARCH=-arch=sm_61

CUDA_FW: single_gpu_FW.cu FW_file_io.cuh FW_file_io.cu
	${COMPILER} -o ${EXE} ${OPT} ${ARCH} single_gpu_FW.cu FW_file_io.cu -lcudart


