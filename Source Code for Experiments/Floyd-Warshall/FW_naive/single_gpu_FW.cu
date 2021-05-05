

/**********************************************************************************

Copyright 2021 Charles W. Johnson
 
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
  
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
**********************************************************************************/


//
// Filename: single_gpu_FW.cpp
// Author: Charles W Johnson
// Description: Single GPU-based Floyd-Warshall algorithm
//
// Note: The GPU kernel and overall approach to implementing Floyd-Warshall like
//       this is from an online example by Saaduddin Mahmud that, as of April 13, 2021,
//       can be found at: https://saadmahmud14.medium.com/parallel-programming-with-cuda-tutorial-part-4-the-floyd-warshall-algorithm-5e1281c46bf6
//
//       The 'packaging' and the rest of the program is all mine (Charles W. Johnson).
//


#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "FW_file_io.cuh"


using namespace std;
using namespace std::chrono;


#define INF 255


/* ---- GPU Kernels ---- */

// Name: FW_IJ_loops_with_next
//
// Description: Naive Floyd-Warshall kernel that computes both the dist and the next matrices
//
__global__ void FW_IJ_loops_with_next(uint8_t* d_dist_pc, int* d_next_pc, uint32_t num_vertices, uint32_t k)
{
    int ij;
    int ik;
    int kj;


    // get the thread's location in the grid
    //
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;

    // get the # of threads in each grid dim
    //
    int x_threads = gridDim.x * blockDim.x;
    int y_threads = gridDim.y * blockDim.y;


    for (int i = x; i < num_vertices; i += x_threads)  // I'm going to start at x,y and then jump
    {                                                  // forward by the x-dim of the grid each time
                                                       // until I can't jump farther

        for (int j = y; j < num_vertices; j += y_threads)  // I'm doing the same thing, only in the y-dim
        {
            // calculate the memory locations of [i][j], [i][k], and [k][j]
            //
            ij = (i * num_vertices) + j;
            ik = (i * num_vertices) + k;
            kj = (k * num_vertices) + j;

            if (d_dist_pc[ij] > (d_dist_pc[ik] + d_dist_pc[kj])) {
                d_dist_pc[ij] = (d_dist_pc[ik] + d_dist_pc[kj]);
                d_next_pc[ij] = d_next_pc[ik];
            }
        } 
    }
}


// Name: FW_IJ_loops
//
// Description: Naive Floyd-Warshall kernel that only computes the dist matrix
//
__global__ void FW_IJ_loops(uint8_t* d_dist_pc, uint32_t num_vertices, uint32_t k)
{
    int ij;
    int ik;
    int kj;


    // get the thread's location in the grid
    //
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;

    // get the # of threads in each grid dim
    //
    int x_threads = gridDim.x * blockDim.x;
    int y_threads = gridDim.y * blockDim.y;


    for (int i = x; i < num_vertices; i += x_threads)  // I'm going to start at x,y and then jump
    {                                                  // forward by the x-dim of the grid each time
                                                       // until I can't jump farther

        for (int j = y; j < num_vertices; j += y_threads)  // I'm doing the same thing, only in the y-dim
        {
            // calculate the memory locations of [i][j], [i][k], and [k][j]
            //
            ij = (i * num_vertices) + j;
            ik = (i * num_vertices) + k;
            kj = (k * num_vertices) + j;

            if (d_dist_pc[ij] > (d_dist_pc[ik] + d_dist_pc[kj])) {
                d_dist_pc[ij] = (d_dist_pc[ik] + d_dist_pc[kj]);
            }
        } 
    }
}


/* ---- CPU functions ---- */

// Name: FW_no_next
//
// Description: Naive Floyd-Warshall function that only computes the dist matrix
//
void FW_no_next(uint8_t** dist, uint32_t num_vertices, double& duration)
{
    auto start = chrono::high_resolution_clock::now();

    for (int k=0; k < num_vertices; k++)
    {
        for (int i=0; i < num_vertices; i++)
        {
            for (int j=0; j < num_vertices; j++)
            {
                if (dist[i][j] > (dist[i][k] + dist[k][j]))
                {
                    dist[i][j] = (dist[i][k] + dist[k][j]);
                }
            }
        }
    }

    auto stop = chrono::high_resolution_clock::now();

    auto temp_duration = duration_cast<microseconds>(stop - start);

    duration = temp_duration.count();
}


// Name: FW_with_next
//
// Description: Naive Floyd-Warshall function that computes both the dist and the next matrices
//
void FW_with_next(uint8_t** dist, int** next, uint32_t num_vertices, double& duration)
{
    auto start = chrono::high_resolution_clock::now();

    for (int k=0; k < num_vertices; k++)
    {
        for (int i=0; i < num_vertices; i++)
        {
            for (int j=0; j < num_vertices; j++)
            {
                if (dist[i][j] > (dist[i][k] + dist[k][j]))
                {
                    dist[i][j] = (dist[i][k] + dist[k][j]);
                    next[i][j] = next[i][k];
                }
            }
        }
    }

    auto stop = chrono::high_resolution_clock::now();

    auto temp_duration = duration_cast<microseconds>(stop - start);

    duration = temp_duration.count();
}




/* ---- The Main Show ---- */


/*

The main() function takes the following arguments:

1 - filename
2 - next ('next' or 'no_next') - this is to determine whether or not a next matrix should be created
3 - save dist ('save_dist' or 'no_save_dist') - this is to decide whether to save a copy of the dist matrix
4 - save next ('save_next' or 'no_save_next') - this is to decide whether to save a copy of the next matrix
5 - console ('console' or 'no_console') - display optional STDOUT
6 - check ('check' or 'no_check') - check the GPU output against CPU output - takes time!!!

*/


int main(int argc, char* argv[]) 
{ 
    // Check to see if we got all arguments
    //
    if (argc < 7) {
        cout << "This function takes the following arguments:" << endl;
        cout << endl;
        cout << "1 - graph filename" << endl;
        cout << "2 - whether or not to create a next matrix - 'next' or 'no_next'" << endl;
        cout << "3 - whether or not to save the dist matrix - 'save_dist' or 'no_save_dist'" << endl;
        cout << "4 - whether or not to save the next matrix - 'save_next' or 'no_save_next'" << endl;
        cout << "5 - whether or not to show the console - 'console' or 'no_console'" << endl;
        cout << "6 - whether or not to run a CPU version and check the GPU output against the CPU output - 'check' or 'no_check'" << endl;
        cout << endl;
        cout << "Warning: The CPU F-W version will take a long time to compute, so only use the check option if you are SURE that you want to do the check!!!" << endl;
        cout << endl;
        return 0;
    }

    // Check to see if the correct values were used and make assignments PRN
    //
    string temp_check = "";

    string next_option = "";
    string save_dist_results = "";
    string save_next_results = "";

    bool show_console = false;
    bool check_output = false;


    // next matrix
    //
    temp_check = argv[2];

    if ((temp_check == "next") || (temp_check == "no_next")) {
        next_option = temp_check;
    } else {
        cout << "The next option must be 'next' or 'no_next'." << endl << endl;
        return 0;
    }

    // Save dist matrix to disk or not
    //
    temp_check = argv[3];

    if ((temp_check == "save_dist") || (temp_check == "no_save_dist")) {
        save_dist_results = temp_check;
    } else {
        cout << "The save dist results option must be 'save_dist' or 'no_save_dist'." << endl << endl;
        return 0;
    }

    // Save next matrix to disk or not
    //
    temp_check = argv[4];

    if ((temp_check == "save_next") || (temp_check == "no_save_next")) {
        save_next_results = temp_check;
    } else {
        cout << "The save next results option must be 'save_next' or 'no_save_next'." << endl << endl;
        return 0;
    }

    // Display certain output to STDOUT
    //
    temp_check = argv[5];

    if ((temp_check == "console") || (temp_check == "no_console")) {
        if (temp_check == "console") {
            show_console = true;
        }
    } else {
        cout << "The show console option must be 'console' or 'no_console'." << endl << endl;
        return 0;
    }


    // Perform a CPU check?
    //
    temp_check = argv[6];

    if ((temp_check == "check") || (temp_check == "no_check")) {
        if (temp_check == "check") {
            check_output = true;
        }
    } else {
        cout << "The check results option must be 'check' or 'no_check'." << endl << endl;
        return 0;
    }


    /* ---- If we get this far, it's now time to read in the num_vertices and num_edges ---- */

    uint32_t num_vertices = 0;           // Number of vertices
    uint32_t num_edges = 0;              // Number of edges

    // Read in the number of vertices from the file
    //
    readInNumVerticesAndEdges(argv[1], num_vertices, num_edges);


    if (show_console) {
        cout << endl;
        cout << "Num of vertices is: " << num_vertices << endl;
        cout << "Num of edges is: " << num_edges << endl;
        cout << endl;
    }


    /* ---- Begin Declaring and Allocating Graph Variables and Memory ---- */

    // -- First, the CPU matrices -- //

    // The host dist matrices

    // h_dist_pc is going to be a huge chunk of memory for the dist matrix.
    // h_dist is going to be how we reference it

    uint64_t mem_size = (sizeof(uint8_t) * num_vertices * num_vertices);

    uint8_t **h_dist;
    uint8_t *h_dist_pc;

    uint8_t **cpu_dist;
    uint8_t *cpu_dist_pc;

    h_dist =    (uint8_t **) malloc(num_vertices * sizeof(uint8_t *));
    h_dist_pc = (uint8_t *)  malloc(mem_size);

    if (check_output)
    {
        cpu_dist =    (uint8_t **) malloc(num_vertices * sizeof(uint8_t *));
        cpu_dist_pc = (uint8_t *)  malloc(mem_size);
    }

if (show_console) cout << "h_dist_pc = " << mem_size << " bytes (" << ((mem_size / 1024.0) / 1024.0) << " MB)" << endl;
if (show_console) cout << endl;

    // Puts a pointer in dist[i] to a place in the chunk
    // of memory that will represent that row.
    //
    for (int i=0; i < num_vertices; i++)
    {
        h_dist[i] = h_dist_pc + (i * num_vertices);

        if (check_output) {
            cpu_dist[i] = cpu_dist_pc + (i * num_vertices);
        }
    }


    // The next matrix

    // h_next_pc is going to be a huge chunk of memory for the h_next matrix.
    // h_next is going to be how we reference it

    int **h_next;
    int *h_next_pc;
    
    int **cpu_next;
    int *cpu_next_pc;
    
    if (next_option == "next")
    {
        mem_size = (sizeof(int) * num_vertices * num_vertices);

        h_next =    (int **) malloc(num_vertices * sizeof(int *));
        h_next_pc = (int *)  malloc(mem_size);

        if (check_output)
        {
            cpu_next =    (int **) malloc(num_vertices * sizeof(int *));
            cpu_next_pc = (int *)  malloc(mem_size);
        }

if (show_console) cout << "h_next_pc = " << mem_size << " bytes (" << ((mem_size / 1024.0) / 1024.0) << " MB)" << endl;
if (show_console) cout << endl;

        // Puts a pointer in h_next[i] to a place in the chunk
        // of memory that will represent that row.
        for (int i=0; i < num_vertices; i++)
        {
            h_next[i] = h_next_pc + (i * num_vertices);

            if (check_output) {
                cpu_next[i] = cpu_next_pc + (i * num_vertices);
            }
        }
    }


    // -- Second, the GPU matrices -- //

    // The dist GPU matrix
    //
    mem_size = (sizeof(uint8_t) * num_vertices * num_vertices);

    uint8_t *d_dist_pc;
    cudaMalloc((void **) &d_dist_pc, (mem_size));

    // The next GPU matrix
    //
    int *d_next_pc;

    if (next_option == "next")
    {
        mem_size = (sizeof(int) * num_vertices * num_vertices);
        cudaMalloc((void **) &d_next_pc, (mem_size));
    }

    // Other GPU variables
    //
    uint32_t *d_num_vertices;
    cudaMalloc((void **) &d_num_vertices, sizeof(uint32_t));

    /* ---- End of Graph Memory Allocation ---- */


    /* ---- Prep the matrices ---- */

    // Initialize all points in the h_dist matrix to INF
    // Initialize all points in the h_next matrix to -1 (PRN)
    //
    for (int i=0; i<num_vertices; i++)
    {
        for (int j=0; j<num_vertices; j++)
        {
            h_dist[i][j] = INF;

            if (check_output) cpu_dist[i][j] = INF;

            if (next_option == "next") {
                h_next[i][j] = -1;

                if (check_output) cpu_next[i][j] = -1;
            }
        }
    }


    /* ---- Read in graph and store the data in the dist matrix ---- */

    bool use_next = false;
    if (next_option == "next") {
        use_next = true;
    }

    readInGraph(argv[1], h_dist, h_next, use_next);

    if (check_output) {
        readInGraph(argv[1], cpu_dist, cpu_next, use_next);
    }


    /* ---- Set the distances for each vertex for itself to be 0 in dist ---- */

    for (int i=0; i<num_vertices; i++)
    {
        h_dist[i][i] = 0;

        if (check_output) cpu_dist[i][i] = 0;

        if (next_option == "next") {
            h_next[i][i] = i;

            if (check_output) cpu_next[i][i] = i;
        }
    }


    /* ---- Copy the data to the GPU ---- */

    auto start_gpu_inc_copy = chrono::high_resolution_clock::now();

    mem_size = (sizeof(uint8_t) * num_vertices * num_vertices);
    cudaMemcpy(d_dist_pc, h_dist_pc, mem_size, cudaMemcpyHostToDevice);

    if (next_option == "next") {
        mem_size = (sizeof(int) * num_vertices * num_vertices);
        cudaMemcpy(d_next_pc, h_next_pc, mem_size, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_num_vertices, &num_vertices, sizeof(uint32_t), cudaMemcpyHostToDevice);


    /* ---- Run the F-W algorithm ---- */

    // As we'll be using grids of 32 threads in the y-axis, then we'll want
    // the num_blocks in the y-axis to be a multiple of 32. And since it's OK
    // if there's a remainder, we can use integer division. However, since there
    // still is the 65,535 limit, we'll look at the number of vertices.

    int blocks;

    if (num_vertices < 65535) {
        if (num_vertices < 32) {
            blocks = 1;
        } else {
            blocks = (num_vertices + 31) / 32;
        }
    } else {

        blocks = 2047;  // the most 32-thread blocks that can fit into 65,535
    }


    dim3 grid(blocks,blocks,1);  // we want to have a square matrix (to keep things simple!)
    dim3 block(32,32,1);         // maximizes occupancy - 1,024 threads/block, so 2 blocks/SM (except for CC 7.5 and 8.0)

    if (show_console) cout << "About to run GPU F-W" << endl << endl;

    auto start_gpu = chrono::high_resolution_clock::now();

    if (next_option == "next")
    {
        for (int k = 0; k < num_vertices; k++)  
        {
            FW_IJ_loops_with_next<<<grid, block>>>(d_dist_pc, d_next_pc, num_vertices, k);
        }

        cudaDeviceSynchronize();
    }
    else
    {
        for (int k = 0; k < num_vertices; k++)  
        {
            FW_IJ_loops<<<grid, block>>>(d_dist_pc, num_vertices, k);
        }

        cudaDeviceSynchronize();
    }

    auto stop_gpu = chrono::high_resolution_clock::now();

    if (show_console) cout << "GPU F-W run complete" << endl << endl;

    /* ---- Copy data back to host ---- */

    mem_size = (sizeof(uint8_t) * num_vertices * num_vertices);
    cudaMemcpy(h_dist_pc, d_dist_pc, mem_size, cudaMemcpyDeviceToHost);

    if (next_option == "next") {
        mem_size = (sizeof(int) * num_vertices * num_vertices);
        cudaMemcpy(h_next_pc, d_next_pc, mem_size, cudaMemcpyDeviceToHost);
    }

    auto stop_gpu_inc_copy = chrono::high_resolution_clock::now();


    /* ---- If specified, run the CPU-based F-W algorithm on cpu_dist (and cpu_next PRN) matrix ---- */

    double duration_cpu = 0;

    if (check_output)
    {
        if (show_console) cout << "About to start CPU F-W run" << endl << endl;

        if (next_option == "next")
        {
            FW_with_next(cpu_dist, cpu_next, num_vertices, duration_cpu);
        }
        else {
            FW_no_next(cpu_dist, num_vertices, duration_cpu);
        }

        if (show_console) cout << "CPU F-W run complete" << endl << endl;
    }


    /* ---- Check the matrices against each other ---- */

    bool check_cleared = true;

    if (check_output)
    {
        if (show_console) cout << "About to start CPU/GPU comparison check" << endl << endl;

        for (int i=0; i < num_vertices; i++)
        {
            for (int j=0; j < num_vertices; j++)
            {
                if (h_dist[i][j] != cpu_dist[i][j]) {
                    check_cleared = false;
                    break;
                }

                if (next_option == "next")
                {
                    if (h_next[i][j] != cpu_next[i][j]) {
                        check_cleared = false;
                        break;
                    }
                }
            }

            if (check_cleared == false) {
                break;
            }
        }

        if (show_console) cout << "CPU/GPU comparison check complete" << endl << endl;
    }

    if (check_output) 
    {
        if (check_cleared == false)
        {
            cout << endl;
            cout << "Check failed - CPU and GPU matrices DO NOT MATCH!" << endl;
            cout << endl;
        } else {
            cout << endl;
            cout << "Check PASSED - CPU and GPU matrices match" << endl;
            cout << endl;
        }
    }


    /* ---- Save the dist and next matrices to disk as called for ---- */

    if (save_dist_results == "save_dist")
    {
        // save the dist matrix to disk
        saveDistMatrixToDisk(argv[1], h_dist, num_vertices, num_edges);
    }

    if ((save_next_results == "save_next") && (next_option == "next"))
    {
        // save the next matrix to disk
        saveNextMatrixToDisk(argv[1], h_next, num_vertices, num_edges);
    }


    /* ---- Print out the runtime for the algorithm itself ---- */

    // Compute the time taken to run algorithm
    auto temp_duration_gpu = duration_cast<microseconds>(stop_gpu - start_gpu);
    auto temp_duration_gpu_inc_copy = duration_cast<microseconds>(stop_gpu_inc_copy - start_gpu_inc_copy);

    double duration_gpu = temp_duration_gpu.count();
    double duration_gpu_inc_copy = temp_duration_gpu_inc_copy.count();


    if (show_console) {
        cout << "Runtime for the GPU F-W algorithm itself is: " << (duration_gpu / 1000.0) << " milliseconds" << endl;
        cout << "Runtime for the GPU F-W algorithm itself is: " << ((duration_gpu / 1000.0) / 1000.0) << " seconds" << endl;
        cout << endl;

        cout << "Runtime for the GPU F-W algorithm with mem copy is: " << (duration_gpu_inc_copy / 1000.0) << " milliseconds" << endl;
        cout << "Runtime for the GPU F-W algorithm with mem copy is: " << ((duration_gpu_inc_copy / 1000.0) / 1000.0) << " seconds" << endl;
        cout << endl;

        if (check_output) {
            cout << "Runtime for the CPU F-W algorithm is: " << (duration_cpu / 1000.0) << " milliseconds" << endl;
            cout << "Runtime for the CPU F-W algorithm is: " << ((duration_cpu / 1000.0) / 1000.0) << " seconds" << endl;
            cout << endl;
        }
    }


    // Free up malloc'ed memory
    //
    free(h_dist);
    free(h_dist_pc);

    if (check_output) {
        free(cpu_dist);
        free(cpu_dist_pc);
    }

    if (next_option == "next") {
        free(h_next);
        free(h_next_pc);

        if (check_output) {
            free(cpu_next);
            free(cpu_next_pc);
        }

        cudaFree(d_next_pc);
    }

    cudaFree(d_dist_pc);
    cudaFree(d_num_vertices);


    // output the result in microseconds
    //
    cout << setprecision(15) << duration_gpu;

    if (show_console) cout << endl;

    return 0; 
} 


