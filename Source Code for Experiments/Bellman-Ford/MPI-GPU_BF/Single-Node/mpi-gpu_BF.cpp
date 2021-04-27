

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
// Filename: mpi-gpu_BF.cpp
// Author: Charles W Johnson
// Description: Distributed GPU-based Bellman-Ford algorithm
//              implementation using MPI
//


#include <chrono>
#include <iostream>

#include "BF_custom_data_structures.h"
#include "BF_file_io.h"
#include "BF_print_functions.h"

#include "BF_kernels.cuh"

#include "mpi.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;


#define INF 255


/* ---- MPI tag definitions ---- */

// for specifying start vertex and # of vertices for a rank/process
//
#define STARTEDGE 101
#define NUMEDGES 102


/* ---- Function Declarations ---- */

// MPI reduction function
//
void dpReduce(void* in, void* inout, int* len, MPI_Datatype* datatype);


// CPU Bellman-Ford implementation
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
// argv[3] = whether or not to print anything, the dists and preds, or the path in the results
//           file - 'no_print', 'no_print_path', or 'print_path'
// argv[4] = whether or not to print the usual console output - 'console' or 'no_console'
// argv[5] = whether or not to run a CPU check - 'check' or 'no_check'
// argv[6] = # of threads/block
// argv[7] = multiple GPUs/node - 'single' or 'multi'
//

int main(int argc, char* argv[]) 
{ 
    /* ---- Check to make sure that we have all arguments and that they are valid values ---- */

    if (argc < 8)
    {
        cout << "This function takes 7 arguments:" << endl;
        cout << endl;
        cout << "1 - The filename for the graph" << endl;
        cout << "2 - Whether to terminate processing once no more changes are being made ('short') or" << endl;
        cout << "    if it should do (num_vertices-1) loops regardless ('normal')" << endl;
        cout << "3 - Whether or not to print anything, the dists and preds, or the path in the results" << endl;
        cout << "    file ('no_print', 'no_print_path', or 'print_path')" << endl;
        cout << "4 - Whether or not to print anything other than the runtime to STDOUT ('console', 'no_console')" << endl;
        cout << "5 - Whether or not to run a CPU check of the results ('check', 'no_check')" << endl;
        cout << "6 - How many threads per block to use in the CUDA grid" << endl;
        cout << "7 - Whether each node has more than one GPU - ('single', 'multi')" << endl;
        cout << endl;
        cout << "Note: The results file will always store the precessor for each vertex, so" << endl;
        cout << "      path reconstruction at a later date is possible. The 'print_path' option" << endl;
        cout << "      specifies whether or not the path is to be drawn out NOW and saved in the" << endl;
        cout << "      results file. Please note: If you choose 'print_path', the results file" << endl;
        cout << "      will be HUGE so only choose that if you have a LOT of disk space available!" << endl;
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

    bool no_print = false;
    bool print_path = false;
    temp_arg = argv[3];

    if ((temp_arg == "no_print") || (temp_arg == "no_print_path") || (temp_arg == "print_path")) {
        if (temp_arg == "no_print") {
            no_print = true;
            print_path = false;
        } else if (temp_arg == "print_path") {
            no_print = false;
            print_path = true;
        }
    } else {
        cout << "You must specify 'no_print', 'no_print_path', or 'print_path'" << endl;
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


    /* ---- Check that tpb > 0 ---- */

    int threadsPerBlock = stoi(argv[6]);

    if ((threadsPerBlock <= 0) || (threadsPerBlock > 1024)) {
        cout << "Threads per block must be above 0 and no higher than 1,024" << endl;
        cout << endl;

        return 0;
    }


    /* ---- Check for single/multi ---- */

    bool multi_gpu = false;
    temp_arg = argv[7];

    if ((temp_arg == "single") || (temp_arg == "multi")) {
        if (temp_arg == "multi") {
            multi_gpu = true;
        }
    } else {
        cout << "You must specify 'single' or 'multi'" << endl;
        cout << endl;

        return 0;
    }


    /* ---------------------------------------------------------------------- */
    /* ---- Get some MPI housekeeping out of the way and get MPI started ---- */
    /* ---------------------------------------------------------------------- */


    /* --- Declare MPI-related variables --- */

    int myid;           // my rank
    int numprocs;       // number of processes
    int rc;             // for function return codes
    MPI_Status status;

    // Just so that we can identify ourselves by more than just our rank
    char hostname[128];
    int hostname_len;


    /* --- Actually get MPI going --- */

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);            // gets us how many processes there are
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);                // gets us our rank
    MPI_Get_processor_name(hostname,&hostname_len);     // gets us our hostname
    hostname[hostname_len] = '\0';


    /* ----------------------------------------------------------------- */
    /* --- We now return you to your regularly scheduled programming --- */
    /* ----------------------------------------------------------------- */


    /* ---- Assign value to some variables that all ranks will need ---- */

    uint32_t source = 0;        // source vertex
    double duration = 0;        // for timing purposes

    uint32_t num_vertices = 0;  // Number of vertices
    uint32_t num_edges = 0;     // Number of edges

    dim3 grid;
    dim3 blocks;

    /* ---- Create data structures all ranks will use ---- */

    // Host data
    //
    Edge* h_edgeList;             // edgeList array
    distPred* h_dp;               // dist/pred array


    // GPU/device data
    //
    Edge* d_edgeList;             // edgeList array
    distPred* d_dp;               // dist/pred array


    /* ---- Memory allocation plan ---- */

    // Because I am currently going to send the entire edgeList to EVERYONE
    // rather than just sending each rank their data and then an offset
    // I'm going to take this approach:
    //
    //  - Rank 0 will read in the num_vertices && num_edges and malloc the
    //    memory that it needs for both host and device arrays.
    //  - Rank 0 will then read the graph data into ITS host arrays and
    //    then copy the data to its GPU.
    //  - Rank 0 will initialize the dp array.
    //  - Rank 0 will (eventually) broadcast the num_vertices, num_edges,
    //    and arrays to all other ranks.
    //
    //  - All other ranks will read in the num_vertices, num_edges, and
    //    malloc the necessary memory for both host and GPU.
    //  - All other ranks will read the arrays into their host arrays
    //    and then copy the data to the GPU.
    //
    //  - After processing is over, outside of the if(myid == ...)
    //    statements, in code that will apply to all ranks, all memory
    //    will be freed.


    /* ---- Create an MPI Datatype to represent the Edge struct ---- */

    // An Edge struct has 2 uint32_ts and 1 uint8_t
    //
    // Creates an MPI datatype called 'mpi_edge'
    //
    int edge_block_length[3] = {1, 1, 1};
    MPI_Datatype edge_type[3] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED_CHAR};
    MPI_Aint edge_disp[3] = {0,4,8};

    MPI_Datatype mpi_edge;
    MPI_Type_create_struct(3, edge_block_length, edge_disp, edge_type, &mpi_edge);
    MPI_Type_commit(&mpi_edge);


    /* ---- Create an MPI Datatype to represent the dist/pred struct ---- */

    // A dist/pred struct has 2 int
    //
    // Creates an MPI datatype called 'dp_struct'
    //
    MPI_Datatype dp_struct;
    MPI_Type_contiguous(2, MPI_INT, &dp_struct);
    MPI_Type_commit(&dp_struct);


    /* ---- Create a custom MPI Reduce operation ---- */

    MPI_Op dpReduceOp;
    MPI_Op_create(dpReduce, 1, &dpReduceOp);


    /* ---- Set GPU # ---- */

    if (multi_gpu) {
        cudaSetDevice(myid);
    } else {
        cudaSetDevice(0);
    }


    /* ---- Some Rank 0 only tasks ---- */

    // Only Rank 0 will read in the full graph
    //
    if (myid == 0)
    {
        if (console)
        {
            cout << endl;
            cout << "You are processing this graph on " << numprocs << " node(s) total." << endl;
            cout << endl;
    
            if (BF_short) {
                cout << "You picked the short route" << endl;
            } else {
                cout << "You picked the normal route" << endl;
            }

            cout << endl;
        }


        /* ---- Get the num_vertices ---- */

        readInNumVerticesAndEdges(argv[1], num_vertices, num_edges);

        if (console)
        {
            cout << "num_vertices = " << num_vertices << endl;
            cout << "num_edges = " << num_edges << endl;
            cout << endl;
        }


        /* ---- Get memory for our data structures ---- */
    
        // Host data
        //
        h_edgeList = (Edge *) malloc(sizeof(Edge) * num_edges);
        h_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);
    
    
        // GPU/device data
        //
        // Note: This cudaMalloc is only done by the master - the slaves will need to do their own
        //
        cudaMalloc((void**) &d_edgeList, (sizeof(Edge) * num_edges));
        cudaMalloc((void**) &d_dp, (sizeof(distPred) * num_vertices));


        /* ---- Only the master reads in the graph ---- */

        readInGraph(argv[1], h_edgeList);


        /* ---- Calculate grid size and set the grid size and block size variables ---- */
    
        int num_blocks = ((num_edges + (threadsPerBlock - 1)) / threadsPerBlock);
    
        grid = dim3(num_blocks, 1, 1);
        blocks = dim3(threadsPerBlock, 1, 1);
    
        if (console) {
            cout << "num_blocks = " << num_blocks << endl;
            cout << "threadsPerBlock = " << threadsPerBlock << endl;
            cout << endl;
        }


        /* ---- Initialize the dp array elements to INF and NULL and set the source vertex ---- */
    
        // By not hard-coding it into the initialization kernel, we can make the source vertex
        // a command-line argument if we want.
        //
        source = 0;

        // Rather than using a for loop, we're going to use the GPU to initialize d_dp. h_dp
        // will then be 'initialized' by copying d_dp from the GPU into h_dp.
        //
        initialize_dp_array_Wrapper(grid, blocks, d_dp, INF, source, num_vertices);

        cudaMemcpy(h_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);
    }


    //////////////////////////////////////////////////////////
    //                                                      //
    // --- Now the work and the MPI stuff really begins --- //
    //                                                      //
    //////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    /* --- This next section is all master node work --- */
    ///////////////////////////////////////////////////////


    /* --- Parse out the edges --- */

    // The plan is that we're going to do basic integer division
    // to get the base num of edges per process. We're then going
    // to do a modulo (edges % proc), so as to learn how many are
    // 'leftover'. We're then going to initialize an array with
    // the base values before going through a loop and adding
    // extra edges to processes until there are no leftovers.

    // These need to be 'global' variables, hence outside of the 'if' statement
    //
    int edges_proc[numprocs];
    int proc_start_edge[numprocs];

    if (myid == 0) {

        /* ---- Print some identifying info to STDOUT ----*/

        if (console)
        {
            // Create a variable for CUDA device properties
            cudaDeviceProp      prop;

            int current_gpu = 0;

            cudaGetDevice(&current_gpu);        // get the device # for the rank's GPU
            cudaGetDeviceProperties(&prop, current_gpu);      // get device data for the rank's GPU

            cout << "My rank is: " << myid << ", my hostname is: " << hostname
                 << ", my GPU device # is:  " << current_gpu << ", and it is a "
                 << prop.name << endl;
            cout << endl;
        }


        /* ---- Now get to work... ---- */

        int start_edge;
        int num_of_edges;
        int num_working_procs;
        int edges_per_proc;
        int leftover_edges;
        int k;

        num_of_edges = num_edges;
        num_working_procs = numprocs;   // all ranks are compute ranks (at least)


        if (console)
        {
            cout << "num_of_edges: " << num_of_edges << endl;
            cout << "num_working_procs: " << num_working_procs << endl;
            cout << endl;
        }

        edges_per_proc = num_of_edges/num_working_procs;
        leftover_edges = num_of_edges%num_working_procs;


        if (console)
        {
            cout << "edges_per_proc: " << edges_per_proc << endl;
            cout << "leftover_edges: " << leftover_edges << endl;
            cout << endl;
        }


        /* ---- Assign edges/proc (or rank) ---- */

        for (k=0; k<numprocs; k++) {
            edges_proc[k] = edges_per_proc;
        }

        k=0;    // reset k to 0

        while (leftover_edges != 0) {    // with this, we're adding to the edge count/load to each rank
            edges_proc[k] += 1;          // until we run out of leftover edges
            leftover_edges--;
            k++;    
        }


        // now we're going to put in an array, the start vertex for each process/rank

        start_edge=0;

        for (k=0; k < numprocs; k++) {
            proc_start_edge[k] = start_edge;
            start_edge += edges_proc[k];
        }


        /* --- Now to send out messages, so as to launch things --- */

        /*
            A quick reference/reminder

            int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
            int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)

            int MPI_Reduce(void *in_buf, void *out_buf, int count, MPI_Datatype input_datatype, operation, int root, MPI_Comm comm)
            int MPI_Allreduce(void *in_buf, void *out_buf, int count, MPI_Datatype input_datatype, operation, MPI_Comm comm)

            int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
        */


        // Declare the control variable for the loop below
        //
        bool finished = false;

        // Send out num_vertices
        MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Send out num_edges
        MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Send out dp
        MPI_Bcast(h_dp, num_vertices, dp_struct, 0, MPI_COMM_WORLD);

        // Send out edgeList
        MPI_Bcast(h_edgeList, num_edges, mpi_edge, 0, MPI_COMM_WORLD);

        // Send out BF_short
        MPI_Bcast(&BF_short, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        // Send out finished
        MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);


        // Send out to each rank its start edge and the number of edges it should process.
        // Note that p begins at 1 - we don't need to tell ourselves our values.
        //
        for (int p=1; p<numprocs; p++)
        {
            // Send a message identifying the start edge, which will be start_edge.
            //
            MPI_Send(&proc_start_edge[p], 1, MPI_INT, p, STARTEDGE, MPI_COMM_WORLD); 

            // Send a message with their number of edges
            //
            MPI_Send(&edges_proc[p], 1, MPI_INT, p, NUMEDGES, MPI_COMM_WORLD); 
        }


        /* ---- Process the edges ---- */

        // We go into a loop here, processing our edges as the other ranks process
        // theirs. We then use an Allreduce on everyone's dp array to get the minimum
        // distances and their predecessors, the function ending with everyone
        // having a copy of the latest dp array. After that, we do an Allreduce
        // on everyone's finished value. Until everyone has a finished value of true,
        // we will continue on with the loop. Finally, we hit an MPI_Barrier() that
        // causes all of the ranks to sync up before continuing, thus ensuring that
        // there won't be any problems with the Allreduces. If finished is true, once
        // all ranks get to the barrier, they will exit the loop.

        // Copy h_edgeList over to GPU - this only needs to be done once
        //
        cudaMemcpy(d_edgeList, h_edgeList, (sizeof(Edge) * num_edges), cudaMemcpyHostToDevice);

        // Now to loop...

        finished = false;  // set finished to false

        // Start the clock
        //
        auto start = chrono::high_resolution_clock::now();

        while (!finished)
        {
            // Copy the latest copy of h_dp to the GPU
            //
            cudaMemcpy(d_dp, h_dp, (sizeof(distPred) * num_vertices), cudaMemcpyHostToDevice);

            // Run Bellman-Ford
            //
            BellmanFord_GPU_Wrapper(0, edges_proc[0], num_vertices, num_edges, d_edgeList, d_dp, BF_short, finished, grid, blocks);

            // Copy the updated copy of d_dp to the CPU (h_dp)
            //
            cudaMemcpy(h_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);

            // Do a reduce on everyone's dp array and a reduce that ends with all of the ranks
            // having the reduced dp array as their dp array.
            //
            MPI_Allreduce(MPI_IN_PLACE, h_dp, num_vertices, dp_struct, dpReduceOp, MPI_COMM_WORLD);

            // Do a reduction on everyone's value of finished. If any rank has finished as false,
            // this should end with the local rank's finished as false. (LAND = logical and)
            //
            MPI_Allreduce(MPI_IN_PLACE, &finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

            // This gets all ranks to sync, for if finished is false and the loop is going to
            // run again, just to be sure that there aren't any mix-ups with the reductions,
            // everyone should start the loop again at the same time.
            //
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Stop the clock
        //
        auto stop = chrono::high_resolution_clock::now();

        // Compute the time taken to run algorithm
        auto temp_duration = duration_cast<microseconds>(stop - start);

        duration = temp_duration.count();

        if (console)
        {
            if (BF_short) {
                cout << "Time taken by Bellman-Ford (Short) loop: " << (duration / 1000.0) << " milliseconds" << endl;
                cout << "Time taken by Bellman-Ford (Short) loop: " << ((duration / 1000.0) / 1000.0) << " seconds" << endl;
                cout << endl;
            } else {
                cout << "Time taken by Bellman-Ford (Normal) loop: " << (duration / 1000.0) << " milliseconds" << endl;
                cout << "Time taken by Bellman-Ford (Normal) loop: " << ((duration / 1000.0) / 1000.0) << " seconds" << endl;
                cout << endl;
            }
        }


        /* ---- Run a CPU check ---- */
    
        if (check_results)
        {
            distPred* check_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);

            double cpu_duration = 0;
    
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


        // If no_print is false, then print a results file, with or without the paths
        //
        if (!no_print)
        {
            if (print_path) {
                writeResultFile(num_vertices, num_edges, h_dp, duration, BF_short);
            } else {
                writeResultFileShort(num_vertices, num_edges, duration, BF_short);
            }
        }

    }   // end Rank 0 if()


    //////////////////////////////////////////////////////
    /* --- This next section is all slave node work --- */
    //////////////////////////////////////////////////////


    // if I'm a slave
    //
    if (myid != 0)
    {
        /* ---- Print some identifying info to STDOUT ----*/

        if (console)
        {
            // Create a variable for CUDA device properties
            cudaDeviceProp      prop;

            int current_gpu = 0;

            cudaGetDevice(&current_gpu);        // get the device # for the rank's GPU
            cudaGetDeviceProperties(&prop, current_gpu);      // get device data for the rank's GPU

            cout << "My rank is: " << myid << ", my hostname is: " << hostname
                 << ", my GPU device # is:  " << current_gpu << ", and it is a "
                 << prop.name << endl;
            cout << endl;
        }


        /* ---- Receive data, etc ---- */

        bool finished = false;

        // Receive num_vertices
        MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Receive num_edges
        MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Get memory for h_dp and d_dp
        //
        h_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);
        cudaMalloc((void**) &d_dp, (sizeof(distPred) * num_vertices));

        // Get memory for h_edgeList and d_edgeList
        //
        h_edgeList = (Edge *) malloc(sizeof(Edge) * num_edges);
        cudaMalloc((void**) &d_edgeList, (sizeof(Edge) * num_edges));

        // Receive dp array
        MPI_Bcast(h_dp, num_vertices, dp_struct, 0, MPI_COMM_WORLD);

        // Receive edgeList
        MPI_Bcast(h_edgeList, num_edges, mpi_edge, 0, MPI_COMM_WORLD);

        // Receive BF_short
        MPI_Bcast(&BF_short, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        // Receive finished
        MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);


        /* ---- Info for THIS rank ---- */

        // Declare the variables for the number of edges to be processed and where in
        // the edgeList to begin.
        //
        int proc_start_edge;
        int proc_num_edges;

        // Receive the starting point
        //
        MPI_Recv(&proc_start_edge, 1, MPI_INT, 0, STARTEDGE, MPI_COMM_WORLD, &status); 

        // Receive the number of edges
        //
        MPI_Recv(&proc_num_edges, 1, MPI_INT, 0, NUMEDGES, MPI_COMM_WORLD, &status);


        // Copy h_edgeList over to GPU - this only needs to be done once
        //
        cudaMemcpy(d_edgeList, h_edgeList, (sizeof(Edge) * num_edges), cudaMemcpyHostToDevice);


        /* ---- Calculate grid size and set the grid size and block size variables ---- */
    
        int num_blocks = ((proc_num_edges + (threadsPerBlock - 1)) / threadsPerBlock);
    
        grid = dim3(num_blocks, 1, 1);
        blocks = dim3(threadsPerBlock, 1, 1);


        // Now to loop...

        finished = false;

        while (!finished)
        {
            // Copy the latest copy of h_dp to the GPU
            //
            cudaMemcpy(d_dp, h_dp, (sizeof(distPred) * num_vertices), cudaMemcpyHostToDevice);

            // Run Bellman-Ford
            //
            BellmanFord_GPU_Wrapper(proc_start_edge, proc_num_edges, num_vertices, num_edges, d_edgeList, d_dp, BF_short, finished, grid, blocks);

            // Copy the updated copy of d_dp to the CPU (h_dp)
            //
            cudaMemcpy(h_dp, d_dp, (sizeof(distPred) * num_vertices), cudaMemcpyDeviceToHost);

            // Do a reduce on everyone's dp array and a reduce that ends with all of the ranks
            // having the reduced dp array as their dp array.
            //
            MPI_Allreduce(MPI_IN_PLACE, h_dp, num_vertices, dp_struct, dpReduceOp, MPI_COMM_WORLD);

            // Do a reduction on everyone's value of finished. If any rank has finished as false,
            // this should end with the local rank's finished as false. (LAND = logical and)
            //
            MPI_Allreduce(MPI_IN_PLACE, &finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

            // This gets all ranks to sync, for if finished is false and the loop is going to
            // run again, just to be sure that there aren't any mix-ups with the reductions,
            // everyone should start the loop again at the same time.
            //
            MPI_Barrier(MPI_COMM_WORLD);
        }

    } // end slave if()


    /* --- Begin housekeeping code for closing things out --- */

    // Free up the dynamically allocated memory

    cudaFree(d_edgeList);
    cudaFree(d_dp);

    free(h_edgeList);
    free(h_dp);

    /* --- Close things MPI out --- */

    MPI_Finalize();


    if (myid == 0) {
        cout << setprecision(15) << duration;  // print the time in MICROseconds to STDOUT
    }

    // If we're printing to STDOUT -and- this is the master rank,, print out a newline so
    // that the runtime # and prompt won't end up on the same line.
    //
    if ((console) && (myid == 0)) {
        cout << endl;
    }

    return 0; 
} 


/* --- MPI Function Definitions --- */


void dpReduce(void* in, void* inout, int* len, MPI_Datatype* datatype)
{
    int A, B;

    for (int i=0; i < *len; i++)
    {
        A = ((distPred *)in)[i].dist;
        B = ((distPred *)inout)[i].dist;

        if (A < B) {
            ((distPred *)inout)[i] = ((distPred *)in)[i];
        }
    }
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


