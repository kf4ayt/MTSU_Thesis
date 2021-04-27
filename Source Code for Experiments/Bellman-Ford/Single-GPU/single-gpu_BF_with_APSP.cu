

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
// Filename: single-gpu_BF.cu
// Author: Charles W Johnson
// Description: Bellman-Ford algorithm using a single GPU
//


#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "BF_custom_data_structures.cuh"
#include "BF_file_io.cuh"
#include "BF_kernels.cuh"
#include "BF_print_functions.cuh"

#define INF 255

using namespace std;
using namespace std::chrono;


/* ---- Support functions ---- */

// Name: BellmanFord
//
// Description: CPU implementation of Bellman-Ford algorithm
//
//
void BellmanFord(uint32_t start_edge, uint32_t num_edges_to_process, uint32_t num_vertices, Edge* edgeList, 
                 distPred* dp, bool BF_short, bool& finished);




/* ---- The Main Show ---- */

//
// Arguments taken by the program:
//
// argv[0] = program name
// argv[1] = graph filename
// argv[2] = which version of BF to use - 'short' or 'normal'
// argv[3] = whether or not to print the path in the output file or print at all - 'print_path', 'no_print_path', or 'no_print'
// argv[4] = whether or not to print the usual console output - 'console' or 'no_console'
// argv[5] = # of threads/block
// argv[6] = 'SSSP' or 'APSP'
//

int main(int argc, char* argv[]) 
{ 
    /* ---- Check to make sure that we have all arguments and that they are valid values ---- */

    if (argc < 7)
    {
        cout << "This function takes 6 arguments:" << endl;
        cout << endl;
        cout << "1 - The filename for the graph" << endl;
        cout << "2 - Whether to terminate processing once no more changes are being made ('short') or" << endl;
        cout << "    if it should do (num_vertices-1) loops regardless ('normal')" << endl;
        cout << "3 - Whether or not to print out the paths in the results file ('print_path', 'no_print_path')" << endl;
        cout << "    or if the results should be printed to a file at all ('no_print')" << endl;
        cout << "4 - Whether or not to print anything other than the runtime to STDOUT ('console', 'no_console')" << endl;
        cout << "5 - How many threads per block to use in the CUDA grid" << endl;
        cout << "6 - 'SSSP' or 'APSP'" << endl;
        cout << endl;
        cout << "Note 1: The results file will always store the precessor for each vertex, so" << endl;
        cout << "        path reconstruction at a later date is possible. The 'print_path' option" << endl;
        cout << "        specifies that the paths be drawn out in the results file NOW." << endl;
        cout << endl;
        cout << "        Please note: If you choose 'print_path', the results file will be HUGE" << endl;
        cout << "        so only choose that if you have a LOT of disk space available!" << endl;
        cout << endl;
        cout << "Note 2: If APSP is selected, the results will NOT be recorded for storage" << endl;
        cout << "        so no results file will/can be printed." << endl;
        cout << endl;

        return 0;
    }

    string temp_arg = "";  // stores an argument value for testing below


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


    /* ---- Check for normal/short ---- */

    bool BF_short = false;
    temp_arg = argv[2];

    if ((temp_arg == "normal") || (temp_arg == "short")) {
        if (temp_arg == "short") {
            BF_short = true;
        }
    } else {
        cout << "You must specify 'normal' or 'short'" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for print_path/no_print_path ---- */

    bool print_path = false;
    bool no_print = false;
    temp_arg = argv[3];

    if ((temp_arg == "print_path") || (temp_arg == "no_print_path") || (temp_arg == "no_print")) {
        if (temp_arg == "print_path") {
            print_path = true;
        }

        if (temp_arg == "no_print") {
            no_print = true;
        }
    } else {
        cout << "You must specify 'print_path', 'no_print_path', or 'no_print'" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for console/no_console ---- */

    bool console = false;
    temp_arg = argv[4];

    if ((temp_arg == "console") || (temp_arg == "no_console")) {
        if (temp_arg == "console") {
            console = true;
        }
    } else {
        cout << "You must specify 'console' or 'no_console'" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check that tpb > 0 ---- */

    int threadsPerBlock = stoi(argv[5]);

    if ((threadsPerBlock <= 0) || (threadsPerBlock > 1024)) {
        cout << "Threads per block must be above 0 and no higher than 1,024" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for SSSP or APSP ---- */

    string graph_operation = "";
    temp_arg = argv[6];

    if ((temp_arg == "SSSP") || (temp_arg == "APSP")) {
        graph_operation = temp_arg;
    } else {
        cout << "You must specify 'SSSP' or 'APSP' for the graph operation" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Declare some variables ---- */

    int source = 0;             // source vertex
    uint32_t num_vertices = 0;  // Number of vertices
    uint32_t num_edges = 0;     // Number of edges

    double gpu_duration = 0;
    double overall_duration = 0;

    // 'change' variables
    //
    int h_change = 0;

    int *d_change = 0;
    cudaMalloc((void**) &d_change, sizeof(int));  // go on and get memory for it


    /* ---- Read in the graph ---- */

    readInNumVerticesAndEdges(argv[1], num_vertices, num_edges);

    if (console) {
        cout << "Read in " << num_vertices << " for num_vertices" << endl;
        cout << "Read in " << num_edges << " for num_edges" << endl;
        cout << endl;
    }


    /* ---- Declare and allocate memory for edgeList and dp arrays ---- */

    // Host (CPU) variables
    //
    Edge* h_edgeList = (Edge *) malloc(sizeof(Edge) * num_edges);
    distPred* h_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);

    h_edgeList = (Edge *) malloc(sizeof(Edge) * num_edges);
    h_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);

    // Device (GPU) variables
    //
    Edge* d_edgeList;
    distPred* d_dp;

    cudaMalloc((void**) &d_edgeList, (sizeof(Edge) * num_edges));
    cudaMalloc((void**) &d_dp, (sizeof(distPred) * num_vertices));


    /* ---- Calculate grid size and set the grid size and block size variables ---- */

    int num_blocks = ((num_edges + (threadsPerBlock - 1)) / threadsPerBlock);

    dim3 grid(num_blocks, 1, 1);
    dim3 threads(threadsPerBlock, 1, 1);

    if (console) {
        cout << "num_blocks = " << num_blocks << endl;
        cout << endl;
    }


    /* ---- Read in the graph ---- */

    readInGraph(argv[1], h_edgeList);


    // By not hard-coding it into the initialization kernel, we can make the source vertex
    // a command-line argument if we want.
    //
    source = 0;


    /* ---- Choose which operation (SSSP or APSP) to perform ---- */

    if (graph_operation == "SSSP")
    {
        if (console) cout << endl << "Executing SSSP on graph, source = " << source << endl << endl;

        /* ---- Initialize the dp array elements to INF and NULL and set the source vertex ---- */

        // Rather than using a for loop, we're going to use the GPU to initialize d_dp. h_dp
        // will not be needed until after all of the processing when we copy d_dp to the host
        // for printing-to-file purposes, so h_dp will never be initialized.
        //
        initialize_dp_array<<<grid, threads>>>(d_dp, INF, source, num_vertices);
    
    
        /* ---- So that we can measure the time needed to copy data to and from the GPU, ---- */
        /* ---- we will start a clock now                                                ---- */
    
        auto overall_start = chrono::high_resolution_clock::now();
    
    
        /* ---- Copy the data to the GPU ---- */
    
        cudaMemcpy(d_edgeList, h_edgeList, (sizeof(Edge) * num_edges), cudaMemcpyHostToDevice);


        /* ---- Run the algorithm ---- */
    
        // Start the clock that's timing the algorithm run itself
        //
        auto start = chrono::high_resolution_clock::now();
    
        for (int i=0; i < (num_vertices-1); i++)
        {
            // we make the local change false
            h_change = 0;
    
            // we copy the local value to the device
            cudaMemcpy(d_change, &h_change, sizeof(int), cudaMemcpyHostToDevice);
    
            // we then run the kernel
            cudaBellmanFord<<<grid, threads>>>(num_edges, d_edgeList, d_dp, d_change);
    
            cudaDeviceSynchronize();
    
            // we now copy the value from the device back to the local variable
            cudaMemcpy(&h_change, d_change, sizeof(int), cudaMemcpyDeviceToHost);
    
            // If the device is reporting a change (h_change == 1), then we are not finished.
            //
            // If the device is reporting that no changes have been made after this pass, then
            // if we are using the 'short' version of B-F, we will cease processing, as there's
            // no point in processing the edges anymore.
            //
            if (BF_short == true) {
                if (!h_change) {
                    break;
                }
            }
        }
    
        // Stop the algorithm run clock
        //
        auto stop = chrono::high_resolution_clock::now();


        /* ---- Copy back the d_dp array to h_dp ---- */
    
        cudaMemcpy(h_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);
    
    
        /* ---- Stop the clock timing the algorithm + mem copies ---- */
    
        auto overall_stop = chrono::high_resolution_clock::now();
    
    
        /* ---- Compute the time taken to run algorithm - w/ and w/o memcpy ---- */
    
        auto temp_gpu_duration = duration_cast<microseconds>(stop - start);
        auto temp_overall_duration = duration_cast<microseconds>(overall_stop - overall_start);
    
        gpu_duration = temp_gpu_duration.count();
        overall_duration = temp_overall_duration.count();

    }  // end SSSP


    if (graph_operation == "APSP")
    {
        if (console) cout << endl << "Executing APSP on graph" << endl << endl;

        source = 0;


        /* ---- Start the clock timing the algorithm + mem copies ---- */
    
        auto overall_start = chrono::high_resolution_clock::now();
    
        while (source < num_vertices)
        {
            /* ---- Initialize the dp array elements to INF and NULL and set the source vertex ---- */
    
            // Rather than using a for loop, we're going to use the GPU to initialize d_dp. h_dp
            // will not be needed until after all of the processing when we copy d_dp to the host
            // for printing-to-file purposes, so h_dp will never be initialized.
            //
            initialize_dp_array<<<grid, threads>>>(d_dp, INF, source, num_vertices);
        
        
            /* ---- Copy the data to the GPU ---- */
        
            cudaMemcpy(d_edgeList, h_edgeList, (sizeof(Edge) * num_edges), cudaMemcpyHostToDevice);
    
    
            /* ---- Run the algorithm ---- */
        
            // Start the clock that's timing the algorithm run itself
            //
            auto start = chrono::high_resolution_clock::now();
        
            for (int i=0; i < (num_vertices-1); i++)
            {
                // we make the local change false
                h_change = 0;
        
                // we copy the local value to the device
                cudaMemcpy(d_change, &h_change, sizeof(int), cudaMemcpyHostToDevice);
        
                // we then run the kernel
                cudaBellmanFord<<<grid, threads>>>(num_edges, d_edgeList, d_dp, d_change);
        
                cudaDeviceSynchronize();
        
                // we now copy the value from the device back to the local variable
                cudaMemcpy(&h_change, d_change, sizeof(int), cudaMemcpyDeviceToHost);
        
                // If the device is reporting a change (h_change == 1), then we are not finished.
                //
                // If the device is reporting that no changes have been made after this pass, then
                // if we are using the 'short' version of B-F, we will cease processing, as there's
                // no point in processing the edges anymore.
                //
                if (BF_short == true) {
                    if (!h_change) {
                        break;
                    }
                }
            }
        
            // Stop the algorithm run clock
            //
            auto stop = chrono::high_resolution_clock::now();


            /* ---- Copy back the d_dp array to h_dp ---- */
    
            cudaMemcpy(h_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);


            // Calculate elapsed time and add total
            //
            auto temp_gpu_duration = duration_cast<microseconds>(stop - start);
            gpu_duration += temp_gpu_duration.count();


            /* ---- If we were to store the data, do it here ---- */


            source++;
        }


        /* ---- Start the clock timing the algorithm + mem copies ---- */
    
        auto overall_stop = chrono::high_resolution_clock::now();
    
        auto temp_overall_duration = duration_cast<microseconds>(overall_stop - overall_start);
        overall_duration = temp_overall_duration.count();

    }  // end APSP


    /* ---- Print to console the times ---- */

    if (console)
    {
        cout << endl;

        // Print the algorithm variant and runtime to the file
        //
        if (BF_short) {
            cout << "B-F processing mode: Short (processing ceases when there are no more changes)" << endl;
        } else {
            cout << "B-F processing mode: Normal (num_vertices-1 loops)" << endl;
        }

        cout << endl;
    
        cout << "Time taken by GPU Bellman-Ford loop: " << (gpu_duration / 1000.0) << " milliseconds" << endl;
        cout << "Time taken by GPU Bellman-Ford loop: " << ((gpu_duration / 1000.0) / 1000.0) << " seconds" << endl;
        cout << endl;
    
        cout << "Time taken including memcpy: " << (overall_duration / 1000.0) << " milliseconds" << endl;
        cout << "Time taken including memcpy: " << ((overall_duration / 1000.0) / 1000.0) << " seconds" << endl;
        cout << endl;
    
        cout << "Difference is: " << ((overall_duration / 1000.0) - (gpu_duration / 1000.0)) << " milliseconds" << endl;
        cout << endl;
    }


    /* ---- Run a CPU check ---- */

    bool runCheck = false;

    if ((graph_operation == "SSSP") && runCheck)
    {
        distPred* check_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);

        for (int a=0; a<num_vertices; a++) {
            check_dp[a].dist = INF;
            check_dp[a].pred = (int)NULL;
        }

        check_dp[source].dist = 0;

        bool finished = false;

        BellmanFord(0, num_edges, num_vertices, h_edgeList, check_dp, BF_short, finished);

        int err_dist_count = 0;
        int err_pred_count = 0;

        for (int i=0; i<num_vertices; i++)
        {
            if (h_dp[i].dist != check_dp[i].dist)
            {
                //cout << "Vertex " << i << "'s dist do not match! - h_dp[i].dist = " << h_dp[i].dist
                //     << ", check_dp[i].dist = " << check_dp[i].dist  << endl;
                err_dist_count++;
            }

            if (h_dp[i].pred != check_dp[i].pred)
            {
                //cout << "Vertex " << i << "'s pred do not match! - h_dp[i].pred = " << h_dp[i].pred
                //     << ", check_dp[i].pred = " << check_dp[i].pred  << endl;
                err_pred_count++;
            }
        }

        cout << endl;
        cout << err_dist_count << " vertices did not match for dist" << endl;
        cout << endl;

        cout << err_pred_count << " vertices did not match for pred" << endl;
        cout << endl;

        free(check_dp);
    }


    /* ---- Write the results to a file ---- */

    if ((!no_print) && (graph_operation == "SSSP")) {
        if (print_path) {
            writeResultFile(num_vertices, num_edges, h_dp, gpu_duration, BF_short);
        } else {
            writeResultFileShort(num_vertices, num_edges, gpu_duration, BF_short);
        }
    }

    /* --- Begin housekeeping code for closing things out --- */

    // Free up the dynamically allocated memory

    free(h_edgeList);
    free(h_dp);

    cudaFree(d_edgeList);
    cudaFree(d_dp);

    cudaFree(d_change);


    // Print to STDOUT the algorithm runtime in milliseconds
    //
    cout << setprecision(15) << gpu_duration;  // print to STDOUT the runtime (exc memcpy) in MICROseconds

    // If we're printing to STDOUT, print out a newline so that the runtime # and prompt
    // won't end up on the same line.
    //
    if (console) {
        cout << endl;
    }

    return 0; 
} 


/* ---- THE Algorithm ---- */

void BellmanFord(uint32_t start_edge, uint32_t num_edges_to_process, uint32_t num_vertices, Edge* edgeList, 
                 distPred* dp, bool BF_short, bool& finished)
{
    int u, v, w;

    bool change = false;
    finished = true;

    for (int i=1; i < (num_vertices-1); i++)
    {
        change = false;

        for (int j=start_edge; j < (start_edge + num_edges_to_process); j++)
        {
            u = edgeList[j].u;
            v = edgeList[j].v;
            w = edgeList[j].w;

            if ((dp[u].dist + w) < dp[v].dist)
            {
                dp[v].dist = (dp[u].dist + w);
                dp[v].pred = u;

                change = true;
            }
        }

        if (change == true) {
            finished = false;
        }

        if (BF_short == true) {
            if (!change) {
                break;
            }
        }
    }
}


