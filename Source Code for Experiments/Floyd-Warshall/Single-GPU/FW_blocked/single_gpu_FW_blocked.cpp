

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
// Filename: single_gpu_FW_tiled.cpp
// Author: Charles W Johnson
// Description: Blocked Single GPU-based Floyd-Warshall algorithm
//
// Note: While I edited the kernels and other associated code so as to allow
//       this program to run inside the framework of my thesis programs, credit
//       for the actual implementation goes to Mateusz Bojanowski. I made enough
//       edits to justify adding the MIT license so as to cover my work, but
//       primary credit goes to Mr. Bojanowski. A copy of his implementation
//       can be found at https://github.com/MTB90/cuda-floyd_warshall as of
//       May 6, 2021.
//


#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "constants.h"


#include "FW_file_io.h"

#include "kernels.cuh"


using namespace std;
using namespace std::chrono;


/* ---- CPU functions ---- */


void FW_no_next(uint8_t** dist, int num_vertices, double& duration);
void FW_with_next(uint8_t** dist, int** next, int num_vertices, double& duration);




/* ---- The Main Show ---- */


/*

The main() function takes the following arguments:

1 - filename
2 - 'next' or 'no_next' - this is to determine whether or not a next matrix should be created
3 - 'no_save', 'save_dist', 'save_dist_and_next' - determines which, if any, output matrices should be saved to disk
4 - 'console' or 'no_console' - display STDOUT
5 - 'check' or 'no_check' - check the GPU output against CPU output - takes time!!!

*/


int main(int argc, char* argv[]) 
{ 
    int num_vertices = 0;
    int num_edges = 0;

    int num_rows = 0;           // number of actual rows/columns in the tiled matrix
    int extra_rows = 0;         // number of extra rows/columns necessary for the matrix to be
                                // of a size that's divisible by 32

    int num_tiles = 0;
    int num_tiles_wide = 0;


    // We're going to explicitly set the GPU to the first one (0)    
    //
    cudaSetDevice(0);


    // Check to see if we got all arguments. Print out a list and bail if we didn't.
    //
    if (argc < 6) {
        cout << "This function takes the following arguments:" << endl;
        cout << endl;
        cout << "1 - graph filename" << endl;
        cout << "2 - 'next' or 'no_next' - this is to determine whether or not a next matrix" << endl;
        cout << "     should be created" << endl;
        cout << "3 - 'no_save', 'save_dist', 'save_dist_and_next' - determines which, if any," << endl;
        cout << "     output matrices should be saved to disk" << endl;
        cout << "4 - 'console' or 'no_console' - display STDOUT" << endl;
        cout << "5 - 'check' or 'no_check' - check the GPU output against CPU output" << endl;
        cout << endl;
        cout << "Warning: The CPU F-W version will take a long time to compute, so only use the check option if you are SURE that you want to do the check!!!" << endl;
        cout << endl;
        return 0;
    }

    // Check to see if the correct values were used and make assignments PRN
    //
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


    /* ---- Need to get num_vertices ---- */

    readInNumVerticesAndEdges(argv[1], num_vertices, num_edges);


    /* ---- Determine size of matrix used for tiled evaluation ---- */

    // If the # of vertices isn't evenly divisible by BLOCK_SIZE...
    //
    if ((num_vertices % BLOCK_SIZE) != 0)
    {
        // Get # of extra vertices (beyond a # divisible by 32)
        //
        extra_rows = (num_vertices % BLOCK_SIZE);

        // Add num_vertices and BLOCK_SIZE -minus- the # of extra rows
        // to get num_rows - the actual matrix size
        //
        num_rows = (num_vertices + (BLOCK_SIZE - extra_rows));
    }
    else
    {
        num_rows = num_vertices;
    }


    /* ---- Calculate size information for tile matrix ---- */

    num_tiles_wide = num_rows / BLOCK_SIZE;
    num_tiles = (num_tiles_wide * num_tiles_wide);


    /* ---- Check for next/no_next ---- */

    bool next = false;

    temp_arg = argv[2];

    if ((temp_arg == "next") || (temp_arg == "no_next")) {
        if (temp_arg == "next") {
            next = true;
        }
    } else {
        cout << "You must specify 'next' or 'no_next'" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for no_save/save_dist/save_dist_and_next ---- */

    bool no_save = false;
    bool save_next = false;
    temp_arg = argv[3];

    if ((temp_arg == "no_save") || (temp_arg == "save_dist") || (temp_arg == "save_dist_and_next")) {

        // This is a check against trying to save a next matrix that doesn't exist.
        // Rather than alerting the user and exiting the program, I've chosen to set
        // temp_arg to "save_dist" as that will save what the user has already indicated
        // that they wish to be saved.
        //
        if ((next == false) && (temp_arg == "save_dist_and_next"))
        {
            temp_arg = "save_dist"; 
        }

        // Determine the values of no_save and save_next
        //
        if (temp_arg == "no_save") {
            no_save = true;
            save_next = false;
        } else if (temp_arg == "save_dist") {
            no_save = false;
            save_next = false;
        } else if (temp_arg == "save_dist_and_next") {
            no_save = false;
            save_next = true;
        }

    } else {
        cout << "You must specify 'no_save', 'save_dist', or 'save_dist_and_next'" << endl;
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


    /* ---- Check for check/no_check ---- */

    bool check_results = false;

    temp_arg = argv[5];

    if ((temp_arg == "check") || (temp_arg == "no_check")) {
        if (temp_arg == "check") {
            check_results = true;
        }
    } else {
        cout << "You must specify 'check' or 'no_check'" << endl;
        cout << endl;

        return 0;
    }


    /* ---- If we get this far, it WOULD BE time to read in the num_vertices and num_edges, ---- */
    /* ---- except for the fact that we've already done so :-)                               ---- */

    // But since we haven't printed them to STDOUT, if 'console' is true...

    if (console) {
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

    uint64_t mem_size = (sizeof(uint8_t) * num_rows * num_rows);

    uint8_t **h_dist;
    uint8_t *h_dist_pc;

    // These have to be declared OUTSIDE the if() statement that gets them memory (if so called for)
    uint8_t **cpu_dist;
    uint8_t *cpu_dist_pc;


    // Get memory for the arrays
    //
    h_dist =    (uint8_t **) malloc(num_rows * sizeof(uint8_t *));
    h_dist_pc = (uint8_t *)  malloc(mem_size);

    if (check_results)
    {
        cpu_dist =    (uint8_t **) malloc(num_rows * sizeof(uint8_t *));
        cpu_dist_pc = (uint8_t *)  malloc(mem_size);
    }

    if (console) cout << "h_dist_pc = " << mem_size << " bytes (" << ((mem_size / 1024.0) / 1024.0) << " MB)" << endl;
    if (console) cout << endl;

    // Puts a pointer in dist[i] to a place in the chunk
    // of memory that will represent that row.
    //
    for (int i=0; i < num_rows; i++)
    {
        h_dist[i] = h_dist_pc + (i * num_rows);

        if (check_results) {
            cpu_dist[i] = cpu_dist_pc + (i * num_rows);
        }
    }


    // The next matrix

    // h_next_pc is going to be a huge chunk of memory for the h_next matrix.
    // h_next is going to be how we reference it

    int **h_next;
    int *h_next_pc;
    
    int **cpu_next;
    int *cpu_next_pc;
    
    if (next)
    {
        mem_size = (sizeof(int) * num_rows * num_rows);

        h_next =    (int **) malloc(num_rows * sizeof(int *));
        h_next_pc = (int *)  malloc(mem_size);

        if (check_results)
        {
            cpu_next =    (int **) malloc(num_rows * sizeof(int *));
            cpu_next_pc = (int *)  malloc(mem_size);
        }

        if (console) cout << "h_next_pc = " << mem_size << " bytes (" << ((mem_size / 1024.0) / 1024.0) << " MB)" << endl;
        if (console) cout << endl;

        // Puts a pointer in h_next[i] to a place in the chunk
        // of memory that will represent that row.
        //
        for (int i=0; i < num_rows; i++)
        {
            h_next[i] = h_next_pc + (i * num_rows);

            if (check_results) {
                cpu_next[i] = cpu_next_pc + (i * num_rows);
            }
        }
    }


    // -- Second, the GPU matrices -- //

    // The dist GPU matrix
    //
    mem_size = (sizeof(uint8_t) * num_rows * num_rows);

    uint8_t *d_dist_pc;
    cudaMalloc((void **) &d_dist_pc, (mem_size));

    // The next GPU matrix
    //
    int *d_next_pc;

    if (next)
    {
        mem_size = (sizeof(int) * num_rows * num_rows);
        cudaMalloc((void **) &d_next_pc, (mem_size));
    }

    // Other GPU variables
    //
    int *d_num_vertices;
    cudaMalloc((void **) &d_num_vertices, sizeof(int));


    /* ---- End of Graph Memory Allocation ---- */


    /* ---- Prep the matrices ---- */

    // Initialize all points in the h_dist matrix to INF
    // Initialize all points in the h_next matrix to -1 (PRN)
    //
    for (int i=0; i<num_rows; i++)
    {
        for (int j=0; j<num_rows; j++)
        {
            h_dist[i][j] = INF;

            if (check_results) cpu_dist[i][j] = INF;

            if (next) {
                h_next[i][j] = -1;

                if (check_results) cpu_next[i][j] = -1;
            }
        }
    }


    /* ---- Read in graph and store the data in the dist matrix ---- */

    bool use_next = false;

    if (next) {
        use_next = true;
    }

    readInGraph(argv[1], h_dist, h_next, use_next);

    if (check_results) {
        readInGraph(argv[1], cpu_dist, cpu_next, use_next);
    }


    // ---- Set the distances for each vertex for itself to be 0 in dist ---- //

    for (int i=0; i<num_rows; i++)
    {
        h_dist[i][i] = 0;

        if (check_results) cpu_dist[i][i] = 0;

        if (next) {
            h_next[i][i] = i;

            if (check_results) cpu_next[i][i] = i;
        }
    }


    /* ---- Copy the data to the GPU ---- */

    mem_size = (sizeof(uint8_t) * num_rows * num_rows);
    cudaMemcpy(d_dist_pc, h_dist_pc, mem_size, cudaMemcpyHostToDevice);

    if (next) {
        mem_size = (sizeof(int) * num_rows * num_rows);
        cudaMemcpy(d_next_pc, h_next_pc, mem_size, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_num_vertices, &num_vertices, sizeof(int), cudaMemcpyHostToDevice);



    /* ---- Run the F-W algorithm ---- */


/*

For the blocked F-W section...

With k running from 0 to (num_rows - 1)...


Step 1
------

Launch a single kernel with a block size of 32x32 (technically, BLOCK_SIZE x BLOCK_SIZE).

In the kernel, first, declare a shared memory array of 32x32 for the dist and another for next.

Next, get the idx - THAT thread's location in the grid. That location will also correspond
to the thread's location in the dist/next matrices.

Then, get id_x and id_y - the thread's coordinates in the block.

Use a combination of num_rows and others to determine if the thread is within the GRAPH - as
opposed to being part of the padding.

Then, load dist[idx] and next[idx] into [id_x][id_y] for the shared dist and next, though
if the thread is in the padding, load INF and -1 into the shared dist and next matrices.

Then, create local variables for the possible dist and next values.

Now, __syncthreads() - we need to hold off proceeding until all threads have loaded their
values into the shared matrices.

The scene set, we now go into a loop from 0 to (i < 32).

 - Step 1 - calculate the new possible dist for that thread.
 - Step 2 - __syncthreads()
 - Step 3 - Compare the possible dist value to the current value and update the values PRN.
 - Step 4 - __syncthreads()
 - Step 5 - Update the shared matrices.

Once we've gone through that, we do another __syncthreads(), update the global matrices with
what's in our position in the shared matrices, and then exit.


Step 2
------

In this step, we run F-W on the k-th row and column.

After going through the steps to fill our own shared memory matrices and find our own locations
as detailed above, we need to create shared memory matrices for the k-th block.


Step 3
------

In this step, we run F-W on all tiles except those already processed


*/


    int tile_k = 0;
    int tile_width = BLOCK_SIZE;

    dim3 grid(1,1,1);                       // we're only launch ONE block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);  // a BLOCK_SIZE of 32 maximizes occupancy - 1,024 threads/block, so 2 blocks/SM (except for CC 7.5 and 8.0)

    tile_k = 0;


    auto start_gpu = chrono::high_resolution_clock::now();

    for (tile_k = 0; tile_k < num_tiles_wide; tile_k++)
    {
        // First, the dependent tile
        //
        grid = dim3(1,1,1);                       // we're only launching ONE block
        block = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);  // a BLOCK_SIZE of 32 maximizes occupancy - 1,024 threads/block, so 2 blocks/SM (except for CC 7.5 and 8.0)

        blocked_Dependent_wrapper(d_dist_pc, d_next_pc, num_vertices, num_rows,
                                  tile_k, tile_width, num_tiles_wide, grid, block);

        cudaDeviceSynchronize();


        // Now, the partially-dependent tiles, starting with the column tiles
        //
        grid = dim3((num_tiles), 1, 1);

        // block variable is the same

        // Note: For the next two kernels, as they could run simultaneously w/o
        //       interfering with each other, there is no cudaDeviceSynchronize
        //       after the Column kernel
        //
        // Process the column
        //
        blocked_Column_wrapper(d_dist_pc, d_next_pc, num_vertices, num_rows,
                               tile_k, num_tiles_wide, tile_width, grid, block);

        // Process the row
        //
        blocked_Row_wrapper(d_dist_pc, d_next_pc, num_vertices, num_rows,
                            tile_k, num_tiles_wide, tile_width, grid, block);

        cudaDeviceSynchronize();


        // Process the independent blocks
        //
        blocked_Independent_wrapper(d_dist_pc, d_next_pc, num_vertices, num_rows,
                                    tile_k, num_tiles_wide, tile_width, grid, block);

        cudaDeviceSynchronize();

    } // end for loop

    auto stop_gpu = chrono::high_resolution_clock::now();


    if (console) {
        cout << "num_tiles = " << num_tiles << endl;
        cout << endl;
    }


    /* ---- Copy data back to host ---- */

    mem_size = (sizeof(uint8_t) * num_rows * num_rows);
    cudaMemcpy(h_dist_pc, d_dist_pc, mem_size, cudaMemcpyDeviceToHost);

    if (next) {
        mem_size = (sizeof(int) * num_rows * num_rows);
        cudaMemcpy(h_next_pc, d_next_pc, mem_size, cudaMemcpyDeviceToHost);
    }


    /* ---- If specified, run the CPU-based F-W algorithm on cpu_dist (and cpu_next PRN) matrix ---- */

    double duration_cpu = 0;

    if (check_results)
    {
        /* ---- Run the CPU F-W algorithm ---- */

        // Note: The algorithm functions take care of the timing and return the runtime,
        //       so we don't have to measure it outside of the function.

        if (console) cout << "About to start CPU F-W run" << endl << endl;

        if (next)
        {
            FW_with_next(cpu_dist, cpu_next, num_vertices, duration_cpu);
        }
        else {
            FW_no_next(cpu_dist, num_vertices, duration_cpu);
        }

        if (console) cout << "CPU F-W run complete" << endl << endl;


        /* ---- Check the matrices against each other ---- */

        bool dist_check_cleared = true;
        bool next_check_cleared = true;

        // ---- dist check ---- //

        for (int i=0; i < num_vertices; i++)
        {
            for (int j=0; j < num_vertices; j++)
            {
                if (h_dist[i][j] != cpu_dist[i][j]) {
                    dist_check_cleared = false;
                    break;
                }
            }

            if (dist_check_cleared == false) {
                break;
            }
        }

        // ---- next check ---- //

        if (next)
        {
            for (int i=0; i < num_vertices; i++)
            {
                for (int j=0; j < num_vertices; j++)
                {
                    if (h_next[i][j] != cpu_next[i][j]) {
                        next_check_cleared = false;
                        break;
                    }
                }
    
                if (next_check_cleared == false) {
                    break;
                }
            }
        }


        /* ---- Output to STDOUT success/failure messages ---- */

        if (dist_check_cleared == false)
        {
            cout << endl;
            cout << "Check failed - CPU and GPU dist matrices DO NOT MATCH!" << endl;
            cout << endl;
        } else {
            cout << endl;
            cout << "Check PASSED - CPU and GPU dist matrices match" << endl;
            cout << endl;
        }

        if (next)
        {
            if (next_check_cleared == false)
            {
                cout << endl;
                cout << "Check failed - CPU and GPU next matrices DO NOT MATCH!" << endl;
                cout << endl;
            } else {
                cout << endl;
                cout << "Check PASSED - CPU and GPU next matrices match" << endl;
                cout << endl;
            }
        }
    }


    /* ---- Save the dist and next matrices to disk as called for ---- */

    if (!no_save)  // if we are going to save something...
    {
        if (save_next)  // if this is true, then we want to save BOTH matrices
        {
            // save the dist matrix to disk
            saveDistMatrixToDisk(argv[1], h_dist, num_vertices, num_edges);

            // save the next matrix to disk
            saveNextMatrixToDisk(argv[1], h_next, num_vertices, num_edges);
        }
        else  // if save_next is false, then we just save the dist matrix
        {
            // save the dist matrix to disk
            saveDistMatrixToDisk(argv[1], h_dist, num_vertices, num_edges);
        }
    }


    /* ---- Print out the runtime for the algorithm itself ---- */

    // Compute the time taken to run algorithm on the GPU (CPU is already computed if necessary)
    //
    auto temp_duration_gpu = duration_cast<microseconds>(stop_gpu - start_gpu);
    double duration_gpu = temp_duration_gpu.count();

    if (console) {
        cout << "Runtime for the GPU F-W algorithm itself is: " << (duration_gpu / 1000.0) << " milliseconds" << endl;
        cout << "Runtime for the GPU F-W algorithm itself is: " << ((duration_gpu / 1000.0) / 1000.0) << " seconds" << endl;
        cout << endl;

        if (check_results) {
            cout << "Runtime for the CPU F-W algorithm is: " << (duration_cpu / 1000.0) << " milliseconds" << endl;
            cout << "Runtime for the CPU F-W algorithm is: " << ((duration_cpu / 1000.0) / 1000.0) << " seconds" << endl;
            cout << endl;
        }
    }


    /* ---- Free up malloc'ed memory ---- */

    free(h_dist);
    free(h_dist_pc);

    if (check_results) {
        free(cpu_dist);
        free(cpu_dist_pc);
    }

    if (next) {
        free(h_next);
        free(h_next_pc);

        if (check_results) {
            free(cpu_next);
            free(cpu_next_pc);
        }

        cudaFree(d_next_pc);
    }

    cudaFree(d_dist_pc);
    cudaFree(d_num_vertices);


    /* ---- output the result in microseconds ---- */

    // This outputs the GPU runtime in microseconds. Unless console or check_results
    // is turned on this will be the program's only output.

    cout << setprecision(15) << duration_gpu;

    if (console) cout << endl;  // a courtesy

    return 0; 
} 




/* ---- CPU Floyd-Warshall functions ---- */


// Runs the F-W algorithm and maintains the next matrix
//
void FW_with_next(uint8_t** dist, int** next, int num_vertices, double& duration_cpu)
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

    auto temp_duration_cpu = duration_cast<microseconds>(stop - start);
    duration_cpu = temp_duration_cpu.count();
}


// Runs the F-W algorithm and does not maintain the next matrix
//
void FW_no_next(uint8_t** dist, int num_vertices, double& duration_cpu)
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

    auto temp_duration_cpu = duration_cast<microseconds>(stop - start);
    duration_cpu = temp_duration_cpu.count();
}


