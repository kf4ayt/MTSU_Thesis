

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
// Filename: basic_Dijkstra_single-GPU.cpp
// Author: Charles W Johnson
// Description: Basic Single-GPU Dijkstra implementation
//


// C++ includes
//
#include <chrono>
#include <iomanip>
#include <iostream>

// Algorithm includes
//
#include "dijkstra-classic-v2.h"
#include "Dijkstra_custom_data_structures.h"
#include "Dijkstra_file_io.h"
#include "Dijkstra_print_functions.h"
#include "Kernels.cuh"

// CUDA includes
//
#include <cuda.h>
#include <cuda_runtime.h>


#define INF 255


using namespace std;
using namespace std::chrono;


void check_results(int* h_V, int* h_E, short int* h_W, distPred* cpu_dp,
                   int num_vertices, int num_edges, int source, distPred* h_dp,
                   double& cpu_duration, bool console);


/* ---- The Big Show ---- */


// Main function takes in the following arguments:
//
// 0 - program name
// 1 - filename for graph file
// 2 - 'SSSP' or 'APSP'
// 3 - 'print_path' or 'no_print_path' - note: for APSP, it won't ever print the paths out
// 4 - threads per block
// 5 - 'console' or 'no_console'
// 6 - 'error_check' or 'no_error_check' - note: only applies to SSSP


int main(int argc, char* argv[]) 
{
    string temp_arg = "";

    /* ---- Make sure that we have all necessary arguments ---- */

    if (argc < 7)
    {
        cout << "This function takes the following arguments:" << endl;
        cout << endl;
        cout << "1 - filename for graph file" << endl;
        cout << "2 - 'SSSP' or 'APSP'" << endl;
        cout << "3 - 'print_path' or 'no_print_path'" << endl;
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

    if ((temp_arg == "print_path") || (temp_arg == "no_print_path")) {

        if (graph_operation == "APSP") {
            printCmd = "no_print_path";
        } else {
            printCmd = temp_arg;
        }
    } else {
        cout << "You must specify 'print_path' or 'no_print_path'" << endl;
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


    /* ------------------------------------------------------------------- */
    /* ---- At this point, we have all of our command line input data ---- */
    /* ------------------------------------------------------------------- */


    /* ---- Declare variables and arrays ---- */

    int num_vertices = 0;
    int num_edges = 0;
    int source = 0;


    // CPU variables
    //
    int* h_V;
    int* h_E;
    short int* h_W;

    distPred* h_dp;
    bool  h_finished;

    distPred* cpu_dp;


    // GPU variables
    //
    int* d_V;
    int* d_E;
    short int* d_W;

    distPred* d_dp;
    bool* d_vertex_settled;
    bool* d_finished;

    dim3 grid;
    dim3 blocks;


    /* ---- Read in the number of vertices and edges ---- */

    readInNumVerticesAndEdges(argv[1], num_vertices, num_edges);


    /* ---- Get CPU and GPU memory ---- */

    // CPU variables
    //
    h_V = (int *) malloc(sizeof(int) * num_vertices);
    h_E = (int *) malloc(sizeof(int) * num_edges);
    h_W = (short int *) malloc(sizeof(short int) * num_edges);

    h_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);
    cpu_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);


    // GPU variables
    //
    cudaMalloc((void**) &d_V, (sizeof(int) * num_vertices));
    cudaMalloc((void**) &d_E, (sizeof(int) * num_edges));
    cudaMalloc((void**) &d_W, (sizeof(short int) * num_edges));

    cudaMalloc((void**) &d_dp, (sizeof(distPred) * num_vertices));
    cudaMalloc((void**) &d_vertex_settled, (sizeof(bool) * num_vertices));
    cudaMalloc((void**) &d_finished, sizeof(bool));


    /* ---- Read in the graph data ---- */

    readInDijkstraGraphData(argv[1], num_vertices, num_edges, h_V, h_E, h_W);


    /* ---- Copy V, E, and W arrays over to the GPU ----*/

    cudaMemcpy(d_V, h_V, (sizeof(int) * num_vertices), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, (sizeof(int) * num_edges), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, (sizeof(short int) * num_edges), cudaMemcpyHostToDevice);


    /* ---- Set values for grid and blocks ---- */

    // Bare minimum value for num_blocks
    //
    int num_blocks = 1;

    // Configure the values to fit the graph unless the graph is
    // small enough to use the default values.
    //
    if (num_vertices > threadsPerBlock)
    {
        num_blocks = (num_vertices + (threadsPerBlock - 1)) / threadsPerBlock;
    }

    // Set grid and blocks values
    //
    grid = dim3(num_blocks, 1, 1);
    blocks = dim3(threadsPerBlock, 1, 1);


    double gpu_duration = 0;

    /* ---- Choose which operation (SSSP or APSP) to perform ---- */

    if (graph_operation == "SSSP")
    {
        int count = 0;

        if (console) cout << endl << "Executing SSSP on graph, source = " << source << endl << endl;

        // Initialize d_dp and d_vertex_settled
        //
        initialize_GPU_arrays_wrapper(d_dp, d_vertex_settled, INF, source, num_vertices, grid, blocks);

        // No cudaDeviceSynchronize needed b/c it's in the wrapper

        // We only need to copy d_dp back into cpu_dp for later error checking. Otherwise, no
        // GPU->CPU copies are necessary.
        //
        cudaMemcpy(cpu_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);

        h_finished = false;


        auto gpu_start = chrono::high_resolution_clock::now();

        while (!h_finished)
        {
            // Set h_finished to false
            //
            h_finished = true;

            // Copy h_finished to d_finished
            //
            cudaMemcpy(d_finished, &h_finished, sizeof(bool), cudaMemcpyHostToDevice);

            // execute the kernel
            //
            Dijkstra_wrapper(d_V, d_E, d_W, d_dp, num_vertices, num_edges,
                             d_vertex_settled, d_finished, grid, blocks);

            // No cudaDeviceSynchronize needed b/c it's in the wrapper

            // Copy d_finished to h_finished
            //
            cudaMemcpy(&h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);

            count++;
        }

        auto gpu_stop = chrono::high_resolution_clock::now();

        // Copy the results back into h_dp
        //
        cudaMemcpy(h_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);

        // Calculate elapsed time
        //
        auto temp_gpu_duration = duration_cast<microseconds>(gpu_stop - gpu_start);
        gpu_duration = temp_gpu_duration.count();

    }  // end SSSP


    if (graph_operation == "APSP")
    {
        int count = 0;

        if (console) cout << endl << "Executing APSP on graph" << endl << endl;

        source = 0;

        while (source < num_vertices)
        {
            // Initialize d_dp and d_vertex_settled
            //
            initialize_GPU_arrays_wrapper(d_dp, d_vertex_settled, INF, source, num_vertices, grid, blocks);

            // No cudaDeviceSynchronize needed b/c it's in the wrapper
    
            // We only need to copy d_dp back into cpu_dp for later error checking. Otherwise, no
            // GPU->CPU copies are necessary.
            //
            // As no error-checking is done for APSP, this cudaMemcpy line is commented out.
            //
            //cudaMemcpy(cpu_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);
    
            h_finished = false;
    
    
            auto gpu_start = chrono::high_resolution_clock::now();
    
            while (!h_finished)
            {
                // Set h_finished to false
                //
                h_finished = true;
    
                // Copy h_finished to d_finished
                //
                cudaMemcpy(d_finished, &h_finished, sizeof(bool), cudaMemcpyHostToDevice);
    
                // execute the kernel
                //
                Dijkstra_wrapper(d_V, d_E, d_W, d_dp, num_vertices, num_edges,
                                 d_vertex_settled, d_finished, grid, blocks);
    
                // No cudaDeviceSynchronize needed b/c it's in the wrapper
    
                // Copy d_finished to h_finished
                //
                cudaMemcpy(&h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
    
                count++;
            }
    
            auto gpu_stop = chrono::high_resolution_clock::now();
    
            // Copy the results back into h_dp
            //
            cudaMemcpy(h_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);

            // Calculate elapsed time and add total
            //
            auto temp_gpu_duration = duration_cast<microseconds>(gpu_stop - gpu_start);
            gpu_duration += temp_gpu_duration.count();


            /* ---- If we were to store the data, do it here ---- */


            source++;
        }

    }  // end APSP


    /* ---- Print total GPU runtime ---- */

    if (console) {
        cout << "GPU time is: " << (gpu_duration / 1000.0) << " ms (does not inc mem copies)" << endl;
        cout << endl;
    }

    /* ---- If running SSSP, this is the optional error-checking ---- */

    if ((graph_operation == "SSSP") && error_check)
    {
        double cpu_duration = 0;

        check_results(h_V, h_E, h_W, cpu_dp, num_vertices, num_edges, source, h_dp, cpu_duration, console);

        if (console) cout << "CPU time is: " << (cpu_duration / 1000.0) << " ms" << endl << endl;
    }


    /* ---- Print results to disk ---- */

    // The APSP section is commented out b/c it is up to the end user as to just
    // how he/she wishes to handle printing APSP results to disk.

    /*
    if (graph_operation == "APSP")
    {
        //printResultsCost(num_vertices, num_edges, h_dp, gpu_duration);
    }
    */

    // SSSP prints costs-only or costs + paths
    //
    if (graph_operation == "SSSP")
    {
        if (printCmd == "print_path") {
            printResultsPath(num_vertices, num_edges, h_dp, gpu_duration);
        } else {
            printResultsCost(num_vertices, num_edges, h_dp, gpu_duration);
        }
    }


    /* ---- Final output ---- */

    cout << setprecision(15) << gpu_duration;

    if (console) cout << endl;


    /* ---- Clean up memory ---- */

    free(h_V);
    free(h_E);
    free(h_W);
    free(h_dp);
    free(cpu_dp);

    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_W);
    cudaFree(d_dp);
    cudaFree(d_vertex_settled);
    cudaFree(d_finished);


    return 0;
}


void check_results(int* h_V, int* h_E, short int* h_W, distPred* cpu_dp,
                   int num_vertices, int num_edges, int source, distPred* h_dp,
                   double& cpu_duration, bool console)
{
    /* ---- Run a CPU implementation to check for errors ---- */

    auto cpu_start = chrono::high_resolution_clock::now();

    dijkstra_classic_cpu(h_V, h_E, h_W, cpu_dp, num_vertices, num_edges, source);

    auto cpu_stop = chrono::high_resolution_clock::now();

    auto temp_cpu_duration = duration_cast<microseconds>(cpu_stop - cpu_start);
    cpu_duration = temp_cpu_duration.count();


    /* ---- Compare the CPU and GPU results ---- */

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

    if (console)
    {
        cout << "Vertex: " << source << endl;
        cout << endl;
        cout << err_dist_count << " dists do not match" << endl;
        cout << endl;
        cout << err_pred_count << " preds do not match" << endl;
        cout << endl;
    }
}


