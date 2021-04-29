

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
// Filename: mpi-cpu_Dijkstra.cpp
// Author: Charles W Johnson
// Description: MPI CPU-based Dijkstra algorithm
//


#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "mpi.h"

#include "Dijkstra_custom_data_structures.h"
#include "Dijkstra_file_io.h"
#include "Dijkstra_print_functions.h"
#include "MinHeap.h"

using namespace std;
using namespace std::chrono;


#define INF 255


/* --- tag definitions --- */

// for specifying start row and # of rows for a rank/process
#define STARTVERTEX 101
#define NUMVERTEX 102


/* ---- Function Declarations ---- */

// THE Algorithm
//
void Dijkstra(uint32_t* V, uint32_t* E, uint8_t* W, distPred* dp, uint32_t num_vertices,
              uint32_t num_edges, int start_vertex, int num_v_to_eval, bool& finished);

// MPI function
//
void dpReduce(void* in, void* inout, int* len, MPI_Datatype* datatype);




/* ---- The Main Show ---- */

//
// Arguments taken by the program:
//
// argv[0] = program name
// argv[1] = graph file name
// argv[2] = whether or not to print anything, the dists and preds, or the path in the results
//           file - 'no_print', 'no_print_path', or 'print_path'
// argv[3] = whether or not to print the usual console output - 'console' or 'no_console'
// argv[4] = whether or not to run a CPU check - 'check' or 'no_check'
//
// Note: The CPU check only outputs anything is the console is set to 'On'
//

int main(int argc, char* argv[]) 
{ 
    /* ---- Check to make sure that we have all arguments and that they are valid values ---- */

    if (argc < 5)
    {
        cout << "This function takes 4 arguments:" << endl;
        cout << endl;
        cout << "1 - The filename for the graph" << endl;
        cout << "2 - Whether or not to print anything, the dists and preds, or the path in the results" << endl;
        cout << "    file ('no_print', 'no_print_path', or 'print_path')" << endl;
        cout << "3 - Whether or not to print anything other than the runtime to STDOUT ('console', 'no_console')" << endl;
        cout << "4 - Whether or not to run a CPU check of the results ('check', 'no_check')" << endl;
        cout << endl;
        cout << "Note: The results file (if opted for) will always store the precessor for each vertex," << endl;
        cout << "      so path reconstruction at a later date is possible. The 'print_path' option" << endl;
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


    /* ---- Check for print_path/print_cost/print_basic ---- */

    bool no_print = false;
    bool print_path = false;
    temp_arg = argv[2];

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
    temp_arg = argv[3];

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
    temp_arg = argv[4];

    if ((temp_arg == "check") || (temp_arg == "no_check")) {
        if (temp_arg == "check") {
            check_results = true;
        }
    } else {
        cout << "You must specify 'check' or 'no_check'" << endl;
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

    int source = 0;             // source vertex
    double duration = 0;        // for timing purposes

    uint32_t num_vertices = 0;  // Number of vertices
    uint32_t num_edges = 0;     // Number of edges


    /* ---- Create data structures all ranks will use ---- */

    uint32_t* V;
    uint32_t* E;
    uint8_t*  W;

    distPred* dp;


    /* ---- Create an MPI Datatype to represent the dist/pred struct ---- */

    // A dist/pred struct is composed of (1) uint8_t and (1) uint32_t, but
    // for the reduction operation to work, we're going with (2) ints.
    //
    // Creates an MPI datatype called 'dp_struct'
    //
    MPI_Datatype dp_struct;
    MPI_Type_contiguous(2, MPI_INT, &dp_struct);
    MPI_Type_commit(&dp_struct);


    /* ---- Create a custom MPI Reduce operation ---- */

    MPI_Op dpReduceOp;
    MPI_Op_create(dpReduce, 1, &dpReduceOp);


    //////////////////////////////////////////////////////////
    //                                                      //
    // --- Now the work and the MPI stuff really begins --- //
    //                                                      //
    //////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////
    /* --- This next section is all master node work --- */
    ///////////////////////////////////////////////////////


    if (myid == 0) {

        /* ---- Read in the number of vertices and edges ---- */

        readInNumVerticesAndEdges(argv[1], num_vertices, num_edges);


        /* ---- Allocate memory for edgeList, dist, and pred arrays ---- */

        // The other ranks will allocate the memory after they have been told
        // what V, etc is

        V = (uint32_t *) malloc(sizeof(uint32_t) * num_vertices);
        E = (uint32_t *) malloc(sizeof(uint32_t) * num_edges);
        W = (uint8_t  *) malloc(sizeof(uint8_t)  * num_edges);

        dp = (distPred *) malloc(sizeof(distPred) * num_vertices);


        /* ---- Only the master reads in the graph ---- */

        readInDijkstraGraphData(argv[1], V, E, W);


        /* ---- Only the master initializes the dp array ---- */

        // The slave ranks will allocate the memory for the arrays, but
        // will populate it with data sent from the master.
        //
        // Note: This may change, as there's no reason why they couldn't
        //       populate the arrays with the initial values themselves.

        // Set the dist and pred arrays to INF and NULL
        //
        for (int a=0; a<num_vertices; a++) {
            dp[a].dist = INF;
            dp[a].pred = (int)NULL;
        }


        /* ---- Set the source vertex's distance ---- */

        // In this case, we're always going to use 0 as the source. We could,
        // though, allow it to be a command line argument, which is why we're
        // going to allow it to be a function argument.
        //
        source = 0;
        dp[source].dist = 0;
    }


    ///////////////////////////////////////////////////////
    /* --- This next section is all master node work --- */
    ///////////////////////////////////////////////////////


    /* --- Parse out the vertices --- */

    // The plan is that we're going to do basic integer division
    // to get the base num of vertices per process. We're then going
    // to do a modulo (vertices % proc), so as to learn how many are
    // 'leftover'. We're then going to initialize an array with
    // the base values before going through a loop and adding
    // extra vertices to processes until there are no leftovers.

    // These need to be 'global' variables, hence outside of the 'if' statement
    //
    int vertices_proc[numprocs];
    int proc_start_vertex[numprocs];

    if (myid == 0)
    {
        /* ---- Now parse out the vertex assignments ---- */

        int start_vertex;
        int num_of_vertices;
        int num_working_procs;
        int vertices_per_proc;
        int leftover_vertices;
        int k;

        num_of_vertices = num_vertices;
        num_working_procs = numprocs;         // all ranks are compute ranks

        if (console) {
            cout << "num_of_vertices: " << num_of_vertices << endl;
            cout << "num_working_procs: " << num_working_procs << endl;
            cout << endl;
        }

        vertices_per_proc = num_of_vertices / num_working_procs;
        leftover_vertices = num_of_vertices % num_working_procs;

        for (k=0; k<numprocs; k++) {
            vertices_proc[k] = vertices_per_proc;
        }

        k=0;    // reset k to 0

        while (leftover_vertices != 0) {    // with this, we're adding to the vertex count/load to each rank
            vertices_proc[k] += 1;          // until we run out of leftover vertex
            leftover_vertices--;
            k++;    
        }


        // now we're going to put in an array, the start vertex for each process/rank
        
        start_vertex=0;

        for (k=0; k < numprocs; k++) {
            proc_start_vertex[k] = start_vertex;
            start_vertex += vertices_proc[k];
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


        // Declare the control value for the loop below
        //
        bool finished = false;


        /* ---- Begin items that will only be sent once ---- */

        // Send out num_vertices 
        MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Send out num_edges 
        MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Send out V
        MPI_Bcast(V, num_vertices, MPI_INT, 0, MPI_COMM_WORLD);

        // Send out E
        MPI_Bcast(E, num_edges, MPI_INT, 0, MPI_COMM_WORLD);

        // Send out W
        MPI_Bcast(W, num_edges, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // Send out dp
        MPI_Bcast(dp, num_vertices, dp_struct, 0, MPI_COMM_WORLD);

        // Send out finished
        MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        /* ---- End items that will only be sent once ---- */


        int p;

        // we don't need to send anything to p=0, as we ARE rank 0
        //
        for (p=1; p<numprocs; p++) {
            // send a message identifying the start vertex, which will be start_vertex.
            MPI_Send(&proc_start_vertex[p], 1, MPI_INT, p, STARTVERTEX, MPI_COMM_WORLD); 

            // send a message with their total vertices
            MPI_Send(&vertices_proc[p], 1, MPI_INT, p, NUMVERTEX, MPI_COMM_WORLD); 
        }


        // Now, we loop - we have broadcasted the graph and sent each slave rank
        // its assigned edges. We now do an MPI_Allreduce reduction on dp[].
        // We follow it with an MPI_Allreduce reduction on finished. If finished
        // is false, then the loop goes through another iteration. If finished is
        // true, then the while loop will exit (on both the master and the slaves).

        finished = false;       // just to be sure that it's set to false

        // Start the clock
        //
        auto start = chrono::high_resolution_clock::now();

        while (!finished)
        {
            // Run Dijkstra
            //
            Dijkstra(V, E, W, dp, num_vertices, num_edges, 0, vertices_proc[0], finished);

            // Do a reduce on everyone's dp array and a reduce that ends with all of the ranks
            // having the reduced dp array as their dp array.
            //
            MPI_Allreduce(MPI_IN_PLACE, dp, num_vertices, dp_struct, dpReduceOp, MPI_COMM_WORLD);

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
            cout << "Time taken by Dijkstra MPI-CPU: " << (duration / 1000.0) << " milliseconds" << endl;
            cout << "Time taken by Dijkstra MPI-CPU: " << ((duration / 1000.0) / 1000.0) << " seconds" << endl;
            cout << endl;
        }


        if (check_results)
        {
            distPred* check_dp = (distPred *) malloc(sizeof(distPred) * num_vertices);

            double cpu_duration = 0;
    
            for (int a=0; a<num_vertices; a++) {
                check_dp[a].dist = INF;
                check_dp[a].pred = (int)NULL;
            }
    
            check_dp[source].dist = 0;
    
            Dijkstra(V, E, W, check_dp, num_vertices, num_edges, 0, num_vertices, finished);
 
            int err_dist_count = 0;
            int err_pred_count = 0;
    
            for (int i=0; i<num_vertices; i++)
            {
                if (dp[i].dist != check_dp[i].dist)
                {
                    //cout << "Vertex " << i << "'s dist do not match! - dp[i].dist = " << dp[i].dist
                    //     << ", check_dp[i].dist = " << check_dp[i].dist << endl;
                    err_dist_count++;
                }
    
                if (dp[i].pred != check_dp[i].pred)
                {
                    //cout << "Vertex " << i << "'s pred do not match!" << endl;
                    //cout << "Vertex " << i << "'s pred do not match! - dp[i].pred = " << dp[i].pred
                    //     << ", check_dp[i].pred = " << check_dp[i].pred << endl;
                    err_pred_count++;
                }
            }

            if (console)
            {
                cout << endl;
                cout << err_dist_count << " vertices did not match for dist" << endl;
                cout << endl;

                cout << err_pred_count << " vertices did not match for pred" << endl;
                cout << endl;
            }
    
            free(check_dp);
        }

        // If no_print is false, then print a results file, with or without the paths
        //
        if (!no_print)
        {
            if (print_path) {
                printResultsPath(num_vertices, num_edges, dp, duration);
            } else {
                printResultsCost(num_vertices, num_edges, dp, duration);
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
        bool finished = false;

        // Receive num_vertices 
        MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Receive num_edges 
        MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // get memory
        //
        V = (uint32_t *) malloc(sizeof(uint32_t) * num_vertices);
        E = (uint32_t *) malloc(sizeof(uint32_t) * num_edges);
        W = (uint8_t  *) malloc(sizeof(uint8_t)  * num_edges);

        dp = (distPred *) malloc(sizeof(distPred) * num_vertices);

        // Receive V
        MPI_Bcast(V, num_vertices, MPI_INT, 0, MPI_COMM_WORLD);

        // Receive E
        MPI_Bcast(E, num_edges, MPI_INT, 0, MPI_COMM_WORLD);

        // Receive W
        MPI_Bcast(W, num_edges, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // Receive dp
        MPI_Bcast(dp, num_vertices, dp_struct, 0, MPI_COMM_WORLD);

        // Receive finished
        MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);


        int proc_start_vertex;
        int num_v_to_eval;

        MPI_Recv(&proc_start_vertex, 1, MPI_INT, 0, STARTVERTEX, MPI_COMM_WORLD, &status); 

        MPI_Recv(&num_v_to_eval, 1, MPI_INT, 0, NUMVERTEX, MPI_COMM_WORLD, &status); 


        while (!finished)
        {
            // Run Dijkstra
            //
            Dijkstra(V, E, W, dp, num_vertices, num_edges, proc_start_vertex, num_v_to_eval, finished);

            // Do a reduce on everyone's dp array and a reduce that ends with all of the ranks
            // having the reduced dp array as their dp array.
            //
            MPI_Allreduce(MPI_IN_PLACE, dp, num_vertices, dp_struct, dpReduceOp, MPI_COMM_WORLD);

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

    free(V);
    free(E);
    free(W);

    free(dp);

    /* --- Close things MPI out --- */

    MPI_Finalize();


    // One last thing...

    if (myid == 0) {
        cout << setprecision(15) << duration;    // print the time in MICROseconds to STDOUT
    }

    if (console) cout << endl;

    return 0; 
} 


/* --- Function Definitions --- */


void Dijkstra(uint32_t* V, uint32_t* E, uint8_t* W, distPred* dp, uint32_t num_vertices, uint32_t num_edges, int start_vertex, int num_v_to_eval, bool& finished)
{
    // declare the size of the heap
    MinHeap Q(num_v_to_eval);   // we only want it as large as the num of vertices that we'll be looking at

    heapNode temp;

    // initialize heap - note that I'm initializing to what's in dp
    //
    for (int a=start_vertex; a<(start_vertex + num_v_to_eval); a++)
    {
        if (a == 0) {
            temp.vertex = a;
            temp.dist = 0;
        } else {
            temp.vertex = a;
            temp.dist = dp[a].dist;
        }

        Q.insertKey(temp);
    }

    // Now for the guts of the program

    int min_vertex = -1;
    int last_edge;
    int temp_v;
    int i;

    bool changed = false;

    finished = true;

    while (Q.heapSize() != 0)
    {
        // The first step is to find the vertex in Q with the smallest distance/cost

        min_vertex = Q.extractMin().vertex;

        // now, min_vertex is the vertex that was in Q that has the min distance/cost

        if (min_vertex == (num_vertices-1))
        {
            last_edge = (num_edges-1);
        } else {
            last_edge = V[min_vertex+1] - 1;
        }

        i = V[min_vertex];      // V[min_vertex] = the first edge, which is stored in i

        while (i <= last_edge)
        {
            temp_v = E[i];      // temp_v contains the vertex at the other end of this edge

            if ((dp[min_vertex].dist + W[i]) < dp[temp_v].dist)
            {
                dp[temp_v].dist = dp[min_vertex].dist + W[i];
                dp[temp_v].pred = min_vertex;

                Q.decreaseKey(temp_v, (dp[min_vertex].dist + W[i]));

                changed = true;
            }

            i++;
        }
    }

    if (changed) {
        finished = false;
    }
}


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


