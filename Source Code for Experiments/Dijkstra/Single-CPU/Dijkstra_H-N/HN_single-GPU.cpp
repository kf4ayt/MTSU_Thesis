

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
 

-------


This additional MIT license was added so as to cover my edits to Harish & Narayanan's
code. In terms of the implementation (the kernels and the control loop, etc), the
changes have been limited to renaming variables, adding necessary wrapper functions,
and then adding support so that the program can keep track of the predecessors,
hence their license remaining on the code. My license is added to cover my edits and
additions, most of which were to allow the program to operate in the framework of
my thesis programs (common file format, etc).

Charles W. Johnson
May, 2021

**********************************************************************************/


/***********************************************************************************
Implementing Single Source Shortest Path on CUDA 1.1 Hardware using algorithm 
given in HiPC'07 paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

Created by Pawan Harish.
************************************************************************************/


/* Begin CWJ includes */

#include <chrono>
#include <iomanip>
#include <iostream>

#include "dijkstra-classic-v2.h"
#include "Dijkstra_custom_data_structures.h"
#include "Dijkstra_file_io.h"
#include "Dijkstra_print_functions.h"

#include "dpKernel.cuh"
#include "H-N_kernels.cuh"

/* End CWJ includes */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>


#define MAX_THREADS_PER_BLOCK 512
#define MAX_COST 10000000


int num_vertices;
int num_edges;


using namespace std;
using namespace std::chrono;


/* ---- Function Declarations ---- */

// Runs the program
//
void DijGraph(int argc, char** argv, int source, string graph_operation, string printCmd,
              int threadsPerBlock, bool console, bool error_check);

// Actually executes the SSSP analysis on the graph
//
void SSSP(int* d_V, int* d_E, short int* d_W, bool* d_graph_mask,
          int num_vertices, int num_edges, distPred* dp, distPred* d_dp_updating_cost,
          bool& changed, bool* d_finished, double& gpu_duration, dim3 grid, dim3 blocks,
          int source);


////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////


// Main function takes in the following arguments:
//
// 0 - program name
// 1 - filename for graph file
// 2 - 'SSSP' or 'APSP'
// 3 - 'no_print', 'print_path', or 'no_print_path' - note: for APSP, it won't ever print the paths out
// 4 - threads per block
// 5 - 'console' or 'no_console'
// 6 - 'error_check' or 'no_error_check' - note: only applies to SSSP


// NOTE: Currently, you can choose whichever option you want to save
//       the results, but none will work as the data-saving code has
//       not been implemented. The functions are compiled into the
//       program, but there is no code below that actually uses them.


int main(int argc, char** argv) 
{
    num_vertices=0;
    num_edges=0;

    int source = 0;

    /* ---- Make sure that we have all necessary arguments ---- */

    if (argc < 7)
    {
        cout << "This function takes the following arguments:" << endl;
        cout << endl;
        cout << "1 - filename for graph file" << endl;
        cout << "2 - 'SSSP' or 'APSP'" << endl;
        cout << "3 - 'no_print', 'print_path', or 'no_print_path'" << endl;
        cout << "     Note: for APSP, it won't ever print the paths out" << endl;
        cout << "4 - threads per block" << endl;
        cout << "5 - 'console' or 'no_console'" << endl;
        cout << "6 - 'error_check' or 'no_error_check'" << endl;
        cout << "     Note: Error checking is only available for SSSP" << endl;

        return 0;
    }


    /* ---- Check that file can be opened ---- */

    ifstream fin;
    fin.open(argv[1]);

    if (fin.fail())
    {
        cout << "You must supply a valid file" << endl;
        cout << endl;
        fin.close();

        return 0;
    } else {
        fin.close();
    }


    string temp_arg = "";


    /* ---- Check that the file is in CSR format (based on the filename) ---- */

    temp_arg = argv[1];
    string graph_format = "";

    size_t result = temp_arg.find("csr", 0);

    if (result != string::npos) {
        graph_format = "csr";
    }

    if (graph_format != "csr")
    {
        cout << "The input file must be in CSR format" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for SSSP or APSP ---- */

    string graph_operation = "";
    temp_arg = argv[2];

    if ((temp_arg == "SSSP") || (temp_arg == "APSP")) {
        graph_operation = temp_arg;
    } else {
        cout << "You must specify 'SSSP' or 'APSP' for the graph operation" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for print_path/no_print_cost ---- */

    string printCmd = "";
    temp_arg = argv[3];

    if ((temp_arg == "no_print") || (temp_arg == "print_path") || (temp_arg == "no_print_path")) {

        if ((graph_operation == "APSP") && (temp_arg == "print_path")) {
            printCmd = "no_print_path";
        } else {
            printCmd = temp_arg;
        }
    } else {
        cout << "You must specify 'no_print', 'print_path', or 'no_print_path'" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for threads per block ---- */

    int threadsPerBlock = stoi(argv[4]);

    if ((threadsPerBlock <= 0) || (threadsPerBlock > 1024)) {
        cout << "Threads per block must be between 1 and 1024" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for console/no_console ---- */

    bool console = false;
    temp_arg = argv[5];

    if ((temp_arg == "console") || (temp_arg == "no_console")) {
        if (temp_arg == "console") {
            console = true;
        }
    } else {
        cout << "You must specify 'console' or 'no_console'" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for error_check/no_error_check ---- */

    bool error_check = false;
    temp_arg = argv[6];

    if ((temp_arg == "error_check") || (temp_arg == "no_error_check")) {
        if (temp_arg == "error_check") {
            error_check = true;
        }
    } else {
        cout << "You must specify 'error_check' or 'no_error_check'" << endl;
        cout << endl;

        return 0;
    }


    // Run the program
    //
    DijGraph(argc, argv, source, graph_operation, printCmd, threadsPerBlock, console, error_check);
 
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//Apply Shortest Path on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////


// Runs the program
//
void DijGraph(int argc, char** argv, int source, string graph_operation, string printCmd,
              int threadsPerBlock, bool console, bool error_check)
{

    auto program_start = chrono::high_resolution_clock::now();

    if (console) printf("Reading File\n");

    // Since the file I/O reads in everything at once, rather than reading in
    // some, pausing to do some work, and then reading in some more, I've had
    // to re-order some of the variable declaration and memory allocation code.
    //
    // First, zero out the num_vertices and num_edges variables;
    //
    num_vertices = 0;
    num_edges = 0;

    // Get the values from the graph
    //
    readInNumVerticesAndEdges(argv[1], num_vertices, num_edges);

    // Declare vertex, edge, and weight arrays
    //
    int* h_V;
    int* h_E;
    short int* h_W;

    // Get the memory for them
    //
    h_V = (int*) malloc(sizeof(int)*num_vertices);
    h_E = (int*) malloc(sizeof(int)*num_edges);
    h_W = (short int*) malloc(sizeof(short int)*num_edges);

    // Read in the graph
    //
    readInDijkstraGraphData(argv[1], num_vertices, num_edges,
                            h_V, h_E, h_W);


    // The graph has been read in. Some output that the original version would give,
    // minus the # of 'dense' nodes:
    //
    if (console) {
        printf("No of Nodes: %d\n", num_vertices);
        printf("Read File\n");
        printf("Avg Branching Factor: %f\n", num_edges/(float)num_vertices);
    }


    // Moving on...

    // Declare and get memory for some other arrays
    //
    bool* h_graph_mask;

    h_graph_mask = (bool*) malloc(sizeof(bool)*num_vertices);


    /* ---- Begin CWJ declarations and memory allocations for the predecessor option ---- */

    // Declare h_dp and d_dp arrays
    //
    distPred* h_dp;
    distPred* h_dp_updating_cost;

    distPred* cpu_dp;

    distPred* d_dp;
    distPred* d_dp_updating_cost;

    // get host memory
    //
    h_dp = (distPred *) malloc(sizeof(distPred)*num_vertices);
    h_dp_updating_cost = (distPred *) malloc(sizeof(distPred)*num_vertices);

    if (error_check) {
         cpu_dp = (distPred *) malloc(sizeof(distPred)*num_vertices);
    }

    // get GPU memory
    //
    cudaMalloc((void**) &d_dp, sizeof(distPred)*num_vertices);
    cudaMalloc((void**) &d_dp_updating_cost, sizeof(distPred)*num_vertices);

    /* ---- End CWJ declarations and memory allocations for the predecessor option ---- */


    /* ---- Declare a lot of GPU arrays, get memory for them, and copy the host array data over ---- */

    // Declare GPU graph arrays
    //
    int* d_V;
    int* d_E;
    short int* d_W;
    bool* d_graph_mask;


    // Get memory for them
    //
    cudaMalloc((void**) &d_V, sizeof(int)*num_vertices);
    cudaMalloc((void**) &d_E, sizeof(int)*num_edges);
    cudaMalloc((void**) &d_W, sizeof(short int)*num_edges);
    cudaMalloc((void**) &d_graph_mask, sizeof(bool)*num_vertices);


    // Copy over h_V, h_E, and h_W
    //
    cudaMemcpy(d_V, h_V, sizeof(int)*num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, sizeof(int)*num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, sizeof(short int)*num_edges, cudaMemcpyHostToDevice);

 
    // Declare a host and GPU variable for finished and get GPU memory for it
    //
    // make a bool to check if the execution is over (H&N comment)
    //
    bool *d_finished;
    bool finished;
    cudaMalloc((void**) &d_finished, sizeof(bool));


    // Configure default values
    //
    int num_of_blocks = 1;
    int num_of_threads_per_block = threadsPerBlock;       // change from = 1

    // Configure the values to fit the graph unless the graph is
    // small enough to use the default values.
    //
    //Make execution Parameters according to the number of nodes (H&N comment)
    //Distribute threads across multiple Blocks if necessary (H&N comment)
    //
    if (num_vertices > threadsPerBlock)
    {
        num_of_blocks = (num_vertices + (threadsPerBlock-1)) / threadsPerBlock;
        num_of_threads_per_block = threadsPerBlock; 
    }

    // Configure grid and threads parameters
    //
    // setup execution parameters (H&N comment)
    //
    dim3  grid(num_of_blocks, 1, 1);
    dim3  blocks(num_of_threads_per_block, 1, 1);


    /* ---- Run SSSP or APSP ---- */

    bool changed = false;

    double gpu_duration = 0;

    auto gpu_start = chrono::high_resolution_clock::now();
    auto gpu_stop = chrono::high_resolution_clock::now();

    if (graph_operation == "SSSP")
    {
        double SSSP_gpu_duration = 0;

        gpu_start = chrono::high_resolution_clock::now();

        // Initialize arrays and make mem copies as noted above
        //
        initialize_dp_array_wrapper(d_dp, d_dp_updating_cost, MAX_COST, source, num_vertices, d_graph_mask, grid, blocks);

        // No cudaDeviceSynchronize needed b/c it's in the wrapper

        cudaMemcpy(h_dp, d_dp, sizeof(distPred)*num_vertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dp_updating_cost, d_dp_updating_cost, sizeof(distPred)*num_vertices, cudaMemcpyDeviceToHost);

        if (error_check) cudaMemcpy(cpu_dp, d_dp, sizeof(distPred)*num_vertices, cudaMemcpyDeviceToHost);

        // Run SSSP
        //
        SSSP(d_V, d_E, d_W, d_graph_mask,
             num_vertices, num_edges, d_dp, d_dp_updating_cost,
             changed, d_finished, SSSP_gpu_duration, grid, blocks, source);

        gpu_stop = chrono::high_resolution_clock::now();

        auto temp_gpu_duration = duration_cast<microseconds>(gpu_stop - gpu_start);
        gpu_duration = temp_gpu_duration.count();


        // Copy the d_dp array back to the host - note: unnecessary for APSP as we're not storing the results
        //
        cudaMemcpy(h_dp, d_dp, sizeof(distPred)*num_vertices, cudaMemcpyDeviceToHost);
    }


    if (graph_operation == "APSP")
    {
        double APSP_gpu_duration = 0;

        // For this, we will be running SSSP for each vertex, so we'll loop over all vertices
        //
        gpu_start = chrono::high_resolution_clock::now();

        for (int i=0; i<num_vertices; i++)
        {
            source = i;
            changed = false;

            // Initialize arrays and make mem copies as noted above
            //
            initialize_dp_array_wrapper(d_dp, d_dp_updating_cost, MAX_COST, source, num_vertices, d_graph_mask, grid, blocks);

            // No cudaDeviceSynchronize needed b/c it's in the wrapper

            // Run SSSP
            //
            SSSP(d_V, d_E, d_W, d_graph_mask,
                 num_vertices, num_edges, d_dp, d_dp_updating_cost,
                 changed, d_finished, APSP_gpu_duration, grid, blocks, source);
        }

        gpu_stop = chrono::high_resolution_clock::now();

        auto temp_gpu_duration = duration_cast<microseconds>(gpu_stop - gpu_start);
        gpu_duration = temp_gpu_duration.count();

        cudaMemcpy(h_dp, d_dp, sizeof(distPred)*num_vertices, cudaMemcpyDeviceToHost);
    }


    auto program_stop = chrono::high_resolution_clock::now();


    /* ---- Run a CPU implementation to check for errors ---- */

    double cpu_duration = 0;

    if (error_check)
    {
        auto cpu_start = chrono::high_resolution_clock::now();

        dijkstra_classic_cpu(h_V, h_E, h_W, cpu_dp, num_vertices, num_edges, source);

        auto cpu_stop = chrono::high_resolution_clock::now();

        auto temp_cpu_duration = duration_cast<microseconds>(cpu_stop - cpu_start);
        cpu_duration = temp_cpu_duration.count();
    }


    /* ---- Compare the CPU and GPU results ---- */

    if (error_check)
    {
        int err_dist_count = 0;
        int err_pred_count = 0;
    
        for (int i=0; i<num_vertices; i++)
        {
            if (cpu_dp[i].dist != h_dp[i].dist)
            {
                //cout << "Vertex " << i << "'s cost don't match! - cpu_dp[i].dist = " << cpu_dp[i].dist
                //     << ", h_dp[i].dist = " << h_dp[i].dist << endl;
    
                err_dist_count++;
            }
    
            if (cpu_dp[i].pred != h_dp[i].pred)
            {
                //cout << "Vertex " << i << "'s preds don't match! - cpu_dp[i].pred = " << cpu_dp[i].pred
                //     << ", h_dp[i].pred = " << h_dp[i].pred << endl;
    
                err_pred_count++;
            }
        }
    
        printf("\n");
        printf("%d vertices have mismatched costs\n", err_dist_count);
        printf("\n");
    
        printf("\n");
        printf("%d vertices have mismatched preds\n", err_pred_count);
        printf("\n");
    }


    /* ---- Print out run times ---- */

    auto temp_program_duration = duration_cast<microseconds>(program_stop - program_start);
    double program_duration = temp_program_duration.count();

    if (console)
    {
        cout << endl;
        cout << "GPU time is: " << (gpu_duration / 1000.0) << "ms" << endl;
        cout << endl;
        cout << "Program time (excluding check) is: " << (program_duration / 1000.0) << "ms" << endl;
        cout << endl;

        if (error_check) {
            cout << "CPU time is: " << (cpu_duration / 1000.0) << "ms" << endl;
            cout << endl;
        }
    }


    /* ---- Print time in MICROseconds to STDOUT ---- */

    cout << setprecision(15) << gpu_duration;

    if (console) cout << endl;


    /* ---- cleanup memory ---- */

    free(h_dp);

    free(h_V);
    free(h_E);
    free(h_W);
    free(h_graph_mask);


    cudaFree(d_dp);

    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_W);
    cudaFree(d_graph_mask);

    cudaFree(d_finished);
}


// Actually executes the SSSP analysis on the graph
//
void SSSP(int* d_V, int* d_E, short int* d_W, bool* d_graph_mask,
          int num_vertices, int num_edges, distPred* d_dp, distPred* d_dp_updating_cost,
          bool& changed, bool* d_finished, double& gpu_duration, dim3 grid, dim3 blocks,
          int source)
{
    // Declare and get memory for finished variables
    //
    bool  h_finished;

    // Start the clock and then start the SSSP loop
    //
    int k=0;

    auto gpu_start = chrono::high_resolution_clock::now();

    do
    {
        DijkstraKernel1_wrapper(d_V, d_E, d_W,
                                d_graph_mask, num_vertices, num_edges,
                                d_dp, d_dp_updating_cost, grid, blocks);
        k++;
        h_finished=false;

        cudaMemcpy(d_finished, &h_finished, sizeof(bool), cudaMemcpyHostToDevice);

        DijkstraKernel2_wrapper(d_V, d_E, d_W,
                                d_graph_mask, d_finished, num_vertices, num_edges,
                                d_dp, d_dp_updating_cost, grid, blocks);

        cudaMemcpy(&h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    while (h_finished);

    // Stop the clock
    //
    auto gpu_stop = chrono::high_resolution_clock::now();

    // Compute the runtime
    //
    auto temp_gpu_duration = duration_cast<microseconds>(gpu_stop - gpu_start);
    gpu_duration = temp_gpu_duration.count();
}


