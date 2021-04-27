#
#
# You will need to edit the INC, LIB, and ARCH values
# to suit your OpenMPI and  CUDA install and hardware.
#
#

#####################################
#### Values used for AWS testing ####
#####################################

#INC=-I/usr/local/cuda-11.0/include -I/opt/amazon/openmpi/include
#LIB=-L/usr/local/cuda-11.0/lib64 -L/opt/amazon/openmpi/lib

#ARCH=-arch=sm_70

#kernel.o: BF_kernels.cu
#	/usr/local/cuda-11.0/bin/nvcc ${INC} ${LIB} ${ARCH} -ccbin /usr/bin/g++ -c BF_kernels.cu -o kernel.o

#####################################


INC=
LIB=

ARCH=-arch=sm_61

all: kernel.o program.o file_io.o print.o
	mpic++ ${INC} ${LIB} kernel.o program.o file_io.o print.o -o MPI-GPU_BF -lcudart

kernel.o: BF_kernels.cu
	/usr/bin/nvcc ${INC} ${LIB} ${ARCH} -ccbin /usr/bin/g++ -c BF_kernels.cu -o kernel.o

program.o: mpi-gpu_BF.cpp
	/usr/bin/mpic++ ${INC} ${LIB} -c mpi-gpu_BF.cpp -o program.o

file_io.o: BF_file_io.cpp
	/usr/bin/mpic++ -c BF_file_io.cpp -o file_io.o

print.o: BF_print_functions.cpp
	/usr/bin/mpic++ -c BF_print_functions.cpp -o print.o

