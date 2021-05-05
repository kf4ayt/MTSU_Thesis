

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
// Filename: mpi-cpu_FW.cpp
// Description: CPU-based MPI version of Floyd-Warshall Algorithm
// Author: Charles W Johnson
//
//
// Note: Credit for this implementation of the Floyd-Warshall algorithm
//       goes to Dr. Stewart Weiss of Hunter College of the City University
//       of New York. There have been some modifications to the implementation,
//       which, as of May 5, 2021, can be found at
//       http://www.compsci.hunter.cuny.edu/~sweiss/course_materials/csci493.65/lecture_notes/chapter05.pdf,
//       but the core implementation is Dr. Weiss'.
// 
//       Changes made by myself are primarily in packaging (file I/O and program
//       arguments) and then in this variation, MPI Rank 0 is the controlling
//       rank for the program. Dr. Weiss' version gives the individual ranks
//       much more authority/responsibility than I have chosen to give them.
//       How the implementation fundamentally works, however, remains the same.
//


#include <chrono>
#include <iostream>

#include "FW_file_io.h"

#include "mpi.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


using namespace std;
using namespace std::chrono;


/* --- tag definitions --- */

// Defines the current K value
//
#define K_VALUE 100
#define K_ROW 101

// for specifying start row and # of rows for a rank/process
//
#define NUM_VERTICES 300
#define NUM_ROWS 301

#define START_ROW 310
#define END_ROW 311

#define DATA 320


#define INF 255



/* --- Declare global problem specific variables --- */

int top_row;            // top row for a rank/process
int bottom_row;         // bottom row for a rank/process


/* --- Function declarations --- */

int findOutRoot(int k, int numprocs, int* proc_start_row, int* rows_proc);




/* --- The Big Show --- */

//
// Arguments taken by the program:
//
// argv[0] = program name
// argv[1] = graph filename
// argv[2] = whether or not to save anything, dist matrix, or the dist and the next matrix
//           in the results file(s) - 'no_save', 'save_dist', or 'save_dist_and_next'
// argv[3] = whether or not to print the usual console output - 'console' or 'no_console'
// argv[4] = whether or not to run a CPU check - 'check' or 'no_check'
//
// Note: The CPU check only outputs anything if the console is set to 'On'
//

int main (int argc, char *argv[]) {

    /* ---- Check to make sure that we have all arguments and that they are valid values ---- */

    if (argc < 5)
    {
        cout << "This function takes 4 arguments:" << endl;
        cout << endl;
        cout << "1 - The filename for the graph" << endl;
        cout << "2 - Whether or not to print anything, dist matrix, or the dist and the next matrix" << endl;
        cout << "    in the results file(s) - 'no_save', 'save_dist', or 'save_dist_and_next'" << endl;
        cout << "3 - Whether or not to print the usual console output - 'console' or 'no_console'" << endl;
        cout << "4 - Whether or not to run a CPU check - 'check' or 'no_check'" << endl;
        cout << endl;
        cout << "Note: Unlike other Floyd-Warshall implementations, this will always calculate a" << endl;
        cout << "      'next' matrix. Whether or not to save the results is up to the user." << endl;
        cout << endl;
        cout << "Note: All results files are stored in binary format. If the 'next' matrix is opted" << endl;
        cout << "      for, it will be stored in a separate file from the 'dist' matrix. In neither" << endl;
        cout << "      file will the paths be drawn out - to do so would result in a HUGE file. If" << endl;
        cout << "      you wish to draw out the paths, the two files will provide all the necessary" << endl;
        cout << "      data for such a task." << endl;
        cout << endl;
        cout << "      For file storage formatting information, see the accompanying README file." << endl;
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


    /* ---- Check for no_print/dist_only/dist_and_next ---- */

    bool no_save = false;
    bool save_next = false;
    temp_arg = argv[2];

    if ((temp_arg == "no_save") || (temp_arg == "save_dist") || (temp_arg == "save_dist_and_next"))
    {
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

    // Just so that we can identify ourselves by more thank just our rank
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


    // Graph details variables
    //
    uint32_t num_vertices = 0;
    uint32_t num_edges = 0;

    double duration;

    int rows_proc[numprocs];
    int proc_start_row[numprocs];


    ///////////////////////////////////////////////////////
    /* --- This next section is all master node work --- */
    ///////////////////////////////////////////////////////


    if (myid == 0)
    {
        // ---- Get the # of vertices and # of edges ---- //

        readInNumVerticesAndEdges(argv[1], num_vertices, num_edges);

        if (console) {
            cout << endl;
            cout << "Overall Graph num_vertices is: " << num_vertices << endl;
            cout << endl;
        }


        /* ---- malloc the memory for the dynamic 2-D array ---- */

        // The dist matrix and dist_temp_row

        uint8_t **dist;
        uint8_t *dist_pc;
        uint8_t *dist_temp_row;

        // dist_pc is going to be a huge chunk of memory for the dist matrix.
        // dist is going to be how we reference it
    
        dist =    (uint8_t **) malloc(sizeof(uint8_t *) * num_vertices);
        dist_pc = (uint8_t *)  malloc(sizeof(uint8_t)   * num_vertices * num_vertices);
    
        // Puts a pointer in dist[i] to a place in the chunk
        // of memory that will represent that row.
        for (int i=0; i < num_vertices; i++)
        {
            dist[i] = dist_pc + (i * num_vertices);
        }


        // The next matrix - NOTE: This stores the VERTICES (so int) not costs!
        //
        // I'm using int so that I can use -1 as NULL

        int **next;
        int *next_pc;
    
        // next_pc is going to be a huge chunk of memory for the next matrix.
        // next is going to be how we reference it
    
        next =    (int **) malloc(sizeof(int *) * num_vertices);
        next_pc = (int *)  malloc(sizeof(int)   * num_vertices * num_vertices);
    
        // Puts a pointer in next[i] to a place in the chunk
        // of memory that will represent that row.
        for (int i=0; i < num_vertices; i++)
        {
            next[i] = next_pc + (i * num_vertices);
        }


        /* ---- Prep the matrices ---- */

        // Initialize all points in the dist matrix to INF and all points
        // in the next matrix to NULL (-1)
        //
        for (int i=0; i<num_vertices; i++)
        {
            for (int j=0; j<num_vertices; j++)
            {
                dist[i][j] = INF;
                next[i][j] = -1;
            }
        }

        // ---- Read in graph and store the data in the dist matrix ---- //
    
        readInGraph(argv[1], dist, next);

    
        // ---- Set the distances for each vertex for itself to be 0 in dist ---- //
        // ---- Set the path for each vertex for itself to be itself in next ---- //
    
        for (int i=0; i<num_vertices; i++)
        {
            dist[i][i] = 0;
            next[i][i] = i;
        }


        // ---- malloc some memory for the 1-D array ---- //

        dist_temp_row = (uint8_t *) malloc(sizeof(uint8_t) * num_vertices);


        // Now that the graph has been read in and is ready for manipulation,
        // we will turn to the administrative part of running this operation.


        // --- Parse out the rows --- //
    
        // The plan is that we're going to do basic integer division
        // to get the base num of rows per process. We're then going
        // to do a modulo (rows % proc), so as to learn how many are
        // 'leftover'. We're then going to initialize an array with
        // the base values before going through a loop and adding
        // extra rows to processes until there are no leftovers.

        int start_row;
        int num_of_rows;
        int num_working_procs;
        int rows_per_proc;
        int leftover_rows;
        int a;


        num_of_rows = num_vertices;         // since we're using an adjacency matrix, these two are equal
        num_working_procs = numprocs;   // this is because Rank #0 is the management rank

        rows_per_proc = num_of_rows/num_working_procs;
        leftover_rows = num_of_rows%num_working_procs;

        // Note that we are starting with a=0 - the master process WILL be number-crunching
        //
        for (a=0; a<numprocs; a++) {
            rows_proc[a] = rows_per_proc;
        }


        a=0;    // reset a to 0

        while (leftover_rows != 0) {    // with this, we're adding to the row count/load to each rank
            rows_proc[a] += 1;          // until we run out of leftover rows
            leftover_rows--;
            a++;    
        }


        // now we're going to put in an array, the start row for each process/rank
        
        a=0;            // reset a to 0 (Rank 0 is INcluded)

        start_row=0;    // make sure start_row is set to 0

        for (a=0; a < numprocs; a++) {
            proc_start_row[a] = start_row;
            start_row += rows_proc[a];
        }


        // The administrative part is complete, so it's now time to start distributing
        // data.

        /*
            A quick reference/reminder

            int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
            int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)

            int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
        */


        uint8_t* dist_start_data;
        int* next_start_data;

        int amt_data;

        // For this, we are starting with a = 1 b/c we don't need to tell ourselves our info
        //
        for (a=1; a<numprocs; a++) {
            // send a message giving the # of vertices
            rc = MPI_Send(&num_vertices, 1, MPI_UNSIGNED, a, NUM_VERTICES, MPI_COMM_WORLD); 

            // send a message identifying the start row, which will be start_row.
            rc = MPI_Send(&proc_start_row[a], 1, MPI_INT, a, START_ROW, MPI_COMM_WORLD); 

            // send a message with their total rows
            rc = MPI_Send(&rows_proc[a], 1, MPI_INT, a, NUM_ROWS, MPI_COMM_WORLD); 

            // send out the data
            //
            amt_data = num_vertices * rows_proc[a];
            dist_start_data = dist_pc + (num_vertices * proc_start_row[a]);
            next_start_data = next_pc + (num_vertices * proc_start_row[a]);

            // send a message with their dist graph data
            rc = MPI_Send(dist_start_data, amt_data, MPI_UNSIGNED_CHAR, a, DATA, MPI_COMM_WORLD); 

            // send a message with their next graph data
            rc = MPI_Send(next_start_data, amt_data, MPI_INT, a, DATA, MPI_COMM_WORLD); 
        }


        /* ---- Go into the K loop ---- */


        int k = 0;
        int k_root = 0;


        auto start = chrono::high_resolution_clock::now();

        for (k=0; k < num_vertices; k++)
        {
            /* ---- These are master-only duties ---- */

            // send out the k number
            MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // discover who 'owns' the k row
            //
            k_root = findOutRoot(k, numprocs, proc_start_row, rows_proc);

            // send out the k_root number
            //
            MPI_Bcast(&k_root, 1, MPI_INT, 0, MPI_COMM_WORLD);


            /* ---- This is where we act as a compute rank ---- */

            // If I'm the rank with the k row, copy it into s_temp_row
            //
            if (myid == k_root)
            {
                int offset = k - 0;

                for (int j=0; j < num_vertices; j++)
                {
                    dist_temp_row[j] = dist[offset][j];
                }

            }

            // Whether we're sending out the k row or receiving it, participate in the
            // broadcast that enables all to send/receive it
            //
            MPI_Bcast(dist_temp_row, num_vertices, MPI_UNSIGNED_CHAR, k_root, MPI_COMM_WORLD);


            /* ---- Run the F-W algorithm on our rows ---- */

            // Note: We are performing it on just OUR chunk of the matrix, so the i loop
            //       is from 0 to OUR number of rows. Also, for the s_dist[k][j]
            //       reference in the algorithm, we are using s_temp_row[j] instead.
            //
            // For the master, i really will start at zero. Otherwise, this is the same
            // as the slave ranks.
            //
            for (int i=0; i < rows_proc[0]; i++)
            {
                for (int j = 0; j < num_vertices; j++)
                {
                    if (dist[i][j] > dist[i][k] + dist_temp_row[j])
                    {
                        dist[i][j] = dist[i][k] + dist_temp_row[j];
                        next[i][j] = next[i][k];
                    }
                }
            }

        }  // end of K loop

        auto stop = chrono::high_resolution_clock::now();


        if (console) {
            cout << endl;
        }


        /* ---- Get the graph data from the slaves ---- */

        for (a=1; a<numprocs; a++)
        {
            // received the data
            //
            amt_data = num_vertices * rows_proc[a];
            dist_start_data = dist_pc + (num_vertices * proc_start_row[a]);
            next_start_data = next_pc + (num_vertices * proc_start_row[a]);

            MPI_Recv(dist_start_data, amt_data, MPI_UNSIGNED_CHAR, a, DATA, MPI_COMM_WORLD, &status);
            MPI_Recv(next_start_data, amt_data, MPI_INT, a, DATA, MPI_COMM_WORLD, &status);
        }


        // ---- Do some admin stuff like print out the time elapsed, etc ---- //

        if (console) {
            cout << "Post F-W alg run..." << endl;
            cout << endl;
        }


        /* ---- single-thread CPU check of MPI results ---- */

        if (check_results)
        {
            /* Get memory, etc */

            uint8_t **check_dist;
            uint8_t *check_dist_pc;
    
            check_dist =    (uint8_t **) malloc(sizeof(uint8_t *) * num_vertices);
            check_dist_pc = (uint8_t *)  malloc(sizeof(uint8_t)   * num_vertices * num_vertices);
        
            for (int i=0; i < num_vertices; i++)
            {
                check_dist[i] = check_dist_pc + (i * num_vertices);
            }
    
    
            int **check_next;
            int *check_next_pc;
        
            check_next =    (int **) malloc(sizeof(int *) * num_vertices);
            check_next_pc = (int *)  malloc(sizeof(int)   * num_vertices * num_vertices);
        
            for (int i=0; i < num_vertices; i++)
            {
                check_next[i] = check_next_pc + (i * num_vertices);
            }


            /* ---- Prep the matrices ---- */
    
            for (int i=0; i<num_vertices; i++)
            {
                for (int j=0; j<num_vertices; j++)
                {
                    check_dist[i][j] = INF;
                    check_next[i][j] = -1;
                }
            }
    
            readInGraph(argv[1], check_dist, check_next);
    
            for (int i=0; i<num_vertices; i++)
            {
                check_dist[i][i] = 0;
                check_next[i][i] = i;
            }


            /* ---- Run the F-W algorithm on it ---- */

            for (int k=0; k < num_vertices; k++)
            {
                for (int i=0; i < num_vertices; i++)
                {
                    for (int j=0; j < num_vertices; j++)
                    {
                        if (check_dist[i][j] > (check_dist[i][k] + check_dist[k][j]))
                        {
                            check_dist[i][j] = (check_dist[i][k] + check_dist[k][j]);
                            check_next[i][j] = check_next[i][k];
                        }
                    }
                }
            }
 

            int err_dist_count = 0;
            int err_next_count = 0;
    
            for (int i=0; i<num_vertices; i++)
            {
                for (int j=0; j<num_vertices; j++)
                {
                    if (dist[i][j] != check_dist[i][j])
                    {
                        //cout << "Vertex " << i << "'s dist do not match! - dp[i].dist = " << dp[i].dist
                        //     << ", check_dp[i].dist = " << check_dp[i].dist << endl;
                        err_dist_count++;
                    }
        
                    if (next[i][j] != check_next[i][j])
                    {
                        //cout << "Vertex " << i << "'s pred do not match!" << endl;
                        //cout << "Vertex " << i << "'s pred do not match! - dp[i].pred = " << dp[i].pred
                        //     << ", check_dp[i].pred = " << check_dp[i].pred << endl;
                        err_next_count++;
                    }
                }
            }

            if (console)
            {
                cout << endl;
                cout << err_dist_count << " cells did not match for dist" << endl;
                cout << endl;

                cout << err_next_count << " cells did not match for next" << endl;
                cout << endl;
            } 

            free(check_dist);
            free(check_dist_pc);
            free(check_next);
            free(check_next_pc);
        }


        /* ---- Save the dist and next matrices to disk as called for ---- */
    
        if (!no_save)  // if we are going to save something...
        {
            if (save_next)  // if this is true, then we want to save BOTH matrices
            {
                // save the dist matrix to disk
                saveDistMatrixToDisk(argv[1], dist, num_vertices, num_edges);
    
                // save the next matrix to disk
                saveNextMatrixToDisk(argv[1], next, num_vertices, num_edges);
            }
            else  // if save_next is false, then we just save the dist matrix
            {
                // save the dist matrix to disk
                saveDistMatrixToDisk(argv[1], dist, num_vertices, num_edges);
            }
        }


        /* ---- Print out the runtime for the MPI-implemented algorithm itself ---- */
    
        // Compute the time taken to run algorithm
        auto temp_duration = duration_cast<microseconds>(stop - start);
    
        duration = temp_duration.count();


        if (console) {
            cout << "Runtime for the MPI F-W algorithm itself is: " << (duration / 1000.0) << " milliseconds" << endl;
            cout << "Runtime for the MPI F-W algorithm itself is: " << ((duration / 1000.0) / 1000.0) << " seconds" << endl;
            cout << endl;
        }


        /* ---- Free memory ---- */

        free(dist);
        free(dist_pc);
        free(next);
        free(next_pc);
        free(dist_temp_row);

    }  // end of master section


    /////////////////////////////////////////////////////////////////
    /* --- From here down, until noted, is just the slave work --- */
    /////////////////////////////////////////////////////////////////

    if (myid > 0) {

        /* ---- Declare vertices and row variables ---- */

        num_vertices = 0;
 
        int slave_start_row;
        int slave_num_rows;


        /* --- Receive num_vertices and row assignments --- */

        // receive a message w/ the # of vertices
        rc = MPI_Recv(&num_vertices, 1, MPI_UNSIGNED, 0, NUM_VERTICES, MPI_COMM_WORLD, &status); 

        // receive a message identifying the start row, which will be slave_start_row.
        rc = MPI_Recv(&slave_start_row, 1, MPI_INT, 0, START_ROW, MPI_COMM_WORLD, &status); 

        // receive a message with the total rows
        rc = MPI_Recv(&slave_num_rows, 1, MPI_INT, 0, NUM_ROWS, MPI_COMM_WORLD, &status); 


        /* ---- Declare variables and get some memory for our graph rows and the temp row ---- */

        uint8_t **s_dist;
        uint8_t *s_dist_pc;
        uint8_t *s_dist_temp_row;
    
        s_dist =    (uint8_t **) malloc(sizeof(uint8_t *) * slave_num_rows);
        s_dist_pc = (uint8_t *)  malloc(sizeof(uint8_t)   * num_vertices * slave_num_rows);

        int **s_next;
        int *s_next_pc;
    
        s_next =    (int **) malloc(sizeof(int *) * slave_num_rows);
        s_next_pc = (int *)  malloc(sizeof(int)   * num_vertices * slave_num_rows);


        // Puts a pointer in s_dist[i] to a place in the chunk of memory that will represent that row.
        //
        for (int i=0; i < slave_num_rows; i++)
        {
            s_dist[i] = s_dist_pc + (i * num_vertices);
        }

        // Puts a pointer in s_next[i] to a place in the chunk of memory that will represent that row.
        //
        for (int i=0; i < slave_num_rows; i++)
        {
            s_next[i] = s_next_pc + (i * num_vertices);
        }


        // malloc for temp_rows
        //
        s_dist_temp_row = (uint8_t *) malloc(sizeof(uint8_t) * num_vertices);


        /* ---- Receive our portion of the graph/matrix ---- */

        MPI_Recv(s_dist_pc, (num_vertices * slave_num_rows), MPI_UNSIGNED_CHAR, 0, DATA, MPI_COMM_WORLD, &status);

        MPI_Recv(s_next_pc, (num_vertices * slave_num_rows), MPI_INT, 0, DATA, MPI_COMM_WORLD, &status);


        /* ---- Declare control variables and go into the K loop ---- */

        int k_root = 0;
        int k_sent = 0;

        for (int k=0; k < num_vertices; k++)
        {
            // receive the k number
            MPI_Bcast(&k_sent, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // receive the k_root number
            MPI_Bcast(&k_root, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // If I'm the rank with the k row, copy it into s_temp_row
            //
            if (myid == k_root)
            {
                int offset = k - slave_start_row;

                for (int j=0; j < num_vertices; j++)
                {
                    s_dist_temp_row[j] = s_dist[offset][j];
                }

            }

            // Whether we're sending out the k row or receiving it, participate in the
            // broadcast that enables all to send/receive it
            //
            MPI_Bcast(s_dist_temp_row, num_vertices, MPI_UNSIGNED_CHAR, k_root, MPI_COMM_WORLD);


            /* ---- Run the F-W algorithm on our rows ---- */

            // Note: We are performing it on just OUR chunk of the matrix, so the i loop
            //       is from 0 to OUR number of rows. Also, for the s_dist[k][j]
            //       reference in the algorithm, we are using s_temp_row[j] instead.
            //
            for (int i=0; i < slave_num_rows; i++)
            {
                for (int j = 0; j < num_vertices; j++)
                {
                    if (s_dist[i][j] > s_dist[i][k] + s_dist_temp_row[j])
                    {
                        s_dist[i][j] = s_dist[i][k] + s_dist_temp_row[j];
                        s_next[i][j] = s_next[i][k];
                    }
                }
            }

        }  // end of K loop


        /* ---- Send data back to the master rank (Rank 0) ---- */

        // For this, we will be in a holding pattern until Rank 0 signals that it wants
        // us to transmit.
        // 
        rc = MPI_Send(s_dist_pc, (num_vertices * slave_num_rows), MPI_UNSIGNED_CHAR, 0, DATA, MPI_COMM_WORLD);

        rc = MPI_Send(s_next_pc, (num_vertices * slave_num_rows), MPI_INT, 0, DATA, MPI_COMM_WORLD);


        /* ---- Free memory ---- */

        free(s_dist);
        free(s_dist_pc);
        free(s_next);
        free(s_next_pc);
        free(s_dist_temp_row);

    }  // End of Slave Section


    ////////////////////////////////////////////////
    //                                            //
    // --- End of Master/Slave code in main() --- //
    //                                            //
    ////////////////////////////////////////////////


    // ---- Close out MPI stuff ---- //

    MPI_Finalize();


    // output the time

    if (myid == 0) {
        cout << setprecision(15) << duration;    // print the time in MICROseconds to STDOUT
    }

    if ((myid == 0) && (console)) {
        cout << endl;
    }

    return 0;
}


/* ---- Function Definitions ---- */


int findOutRoot(int k, int numprocs, int* proc_start_row, int* rows_proc)
{
    int k_root = 0;
    int start_row = 0;
    int num_rows = 0;
    int end_row = 0;

    for (int a=1; a < numprocs; a++)
    {
        start_row = proc_start_row[a];
        num_rows = rows_proc[a];
        end_row = start_row + (num_rows - 1); 

        if ((k >= start_row) && (k <= end_row))
        {
            k_root = a;
            break;
        }
    }

    return k_root;
}


