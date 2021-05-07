

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
// Filename: kernels.cu
// Author: Charles W Johnson
// Description: Kernels for single GPU-based Floyd-Warshall algorithm
//
// Note: While I edited the kernels and other associated code so as to allow
//       this program to run inside the framework of my thesis programs, credit
//       for the actual implementation goes to Mateusz Bojanowski. I made enough
//       edits to justify adding the MIT license so as to cover my work, but
//       primary credit goes to Mr. Bojanowski. A copy of his implementation
//       can be found at https://github.com/MTB90/cuda-floyd_warshall as of
//       May 6, 2021.
//


#include "kernels.cuh"


using namespace std;
using namespace std::chrono;


/* ---- CPU Wrapper Functions ---- */

void blocked_Dependent_wrapper(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                               int tile_k, int tile_width, int num_tiles_wide, dim3 grid, dim3 blocks)
{
    blocked_Dependent<<<grid, blocks>>>(d_dist_pc, d_next_pc, num_vertices, num_rows,
                                       tile_k, tile_width, num_tiles_wide);
}


void blocked_Column_wrapper(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                            int tile_k, int num_tiles_wide, int tile_width, dim3 grid, dim3 blocks)
{
    blocked_Column<<<grid, blocks>>>(d_dist_pc, d_next_pc, num_vertices, num_rows, tile_k, num_tiles_wide, tile_width);
}


void blocked_Row_wrapper(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                         int tile_k, int num_tiles_wide, int tile_width, dim3 grid, dim3 blocks)
{
    blocked_Row<<<grid, blocks>>>(d_dist_pc, d_next_pc, num_vertices, num_rows, tile_k, num_tiles_wide, tile_width);
}


void blocked_Independent_wrapper(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                                 int tile_k, int num_tiles_wide, int tile_width, dim3 grid, dim3 blocks)
{
    blocked_Independent<<<grid, blocks>>>(d_dist_pc, d_next_pc, num_vertices, num_rows, tile_k, num_tiles_wide, tile_width);
}




/* ---- GPU Kernels ---- */

__global__ void blocked_Dependent(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                                 int tile_k, int tile_width, int num_tiles_wide)
{
    // Declare the shared memory matrices that this thread and the others in its block will
    // use when performing the F-W calculations.
    //
    __shared__ uint8_t shared_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_next[BLOCK_SIZE][BLOCK_SIZE];


    // We now need four sets of coordinates - 1) the location of the thread in the block,
    // 2) the location of our thread in d_dist_pc/d_next_pc, 3) the thread's location
    // within the graph/matrix as a whole, expressed in x/y values, and 4) the location of
    // the block it is in. For computing the dependent block, this is not relevent, but for
    // computing the others, it will be.

    /* ---- 1st set of coordinates ---- */

    // The thread's x/y coordinates for its position in the block, though expressed as i,j
    //
    int i = threadIdx.x;
    int j = threadIdx.y;


    /* ---- 2nd set of coordinates ---- */

// update

    // To determine memory location, the basic formula for a grid is for location (x,y), we
    // calc (y * width_of_rows + x). In this case, because we are operating on tiles, it is
    // a little trickier, for if we have a 90-vertex graph, thus giving us a graph matrix of
    // 90x90, with a 32x32 block, we would have (3) blocks for a width of 96. The row width,
    // however, would still be 90. Thus, to get the correct location in memory - d_dist_pc
    // and d_next_pc - we need to use num_vertices as our row width, NOT the total width
    // of the TILE matrix. Thus, 'width_of_rows' will be num_vertices. Since we are dealing
    // with tiles here, however, the chunk of memory that would normally be 1 row is actually
    // going to be tile_width * num_vertices.
    //

    int tile_row_width = tile_width * num_tiles_wide;

    // To reach the k-th dependent block, we multiply our row_width by tile_k.
    //
    int row_mem_start = tile_k * (tile_row_width * tile_width);

    // At this point, in memory, we are at the start of the k-th TILE row. Since j will
    // apply regardless of the i value, we multiply (j * row_width) to get how much
    // farther ahead we need to move to get to the correct graph row. For the i value,
    // since we know what the current k value is (tile_k) and know the tile_width, then
    // (tile_k * tile_width) will get us to the beginning of the desired block. Adding
    // i to that will finish the process.
    //
    int thread_mem_loc = (j * tile_row_width) + (tile_k * tile_width) + i;

    // idx = row_mem_start + thread_mem_loc - the start of the TILE row + the amount to move
    // forward in the graph itself to get to the correct graph (i,j) position.
    //
    int idx = row_mem_start + thread_mem_loc;


    /* ---- 3rd set of coordinates ---- */

    // To determine the thread's graph_i/graph_j coordinates, we can get graph_j by multiplying
    // the number of tiles * tile_width and then adding j (our position inside the block).
    //
    int graph_j = (tile_k * tile_width) + j;

    // To get graph_i, we multiply tile_k * tile_width and then add i to that.
    //
    int graph_i = (tile_k * tile_width) + i;

    // Note: The above only works because we are working on the dependent block where the number
    //       of tiles in each axis is the same. Furthermore, that does not guarantee that our
    //       thread is actually in the graph (which is why there are later checks).


    /* ---- 4th set of coordinates ---- */

    // This determines where our block is in the TILE matrix. If its tile_i and tile_j values
    // don't equal tile_k, we bail.
    //
    // Note: For this kernel, the code will not be included as it would be pointless - this
    //       kernel is only launched with one block.


    /* ---- Load our data into the shared memory matrixes ---- */

    // Since it's possible that the graph will not fully occupy the block, then we need to
    // determine whether or not this thread is actually representing an element in the graph.
    // If it is not, we'll set the dist and next values such that it won't affect the proper
    // processing of the graph.

    // Determine whether or not the thread is in the actual graph or just padding.
    //
    bool in_graph = false;

    if ((graph_i < num_vertices) && (graph_j < num_vertices)) {
        in_graph = true;
    }


    // Load the thread's dist and next values into its shared memory location.
    //
    if (in_graph) {
        shared_dist[i][j] = d_dist_pc[idx];
        shared_next[i][j] = d_next_pc[idx];
    } else {
        shared_dist[i][j] = INF;
        shared_next[i][j] = -1;
    }

    // Stop and wait until everyone has loaded their data into the shared matrices
    //
    __syncthreads();


    /* ---- Now to run the F-W algorithm ---- */

    // Temp variables for possible new values, initialized to starting values
    //
    int pos_Dist = shared_dist[i][j];
    int pos_Next = shared_next[i][j];

    // The loop
    //
    for (int k=0; k < BLOCK_SIZE; k++)
    {

        //pos_Dist = tile_k; 
        //pos_Dist = 0; 

        //shared_dist[i][j] = pos_Dist;

        //if (shared_dist[i][j] == 255) {
        //    shared_dist[i][j] = -1;
        //} else {
        //    shared_dist[i][j] = shared_dist[i][j] + 1;
        //}

        //shared_dist[i][j] = shared_dist[i][j];


        pos_Dist = shared_dist[i][k] + shared_dist[k][j]; 

        __syncthreads();

        if (pos_Dist < shared_dist[i][j])
        {
            shared_dist[i][j] = pos_Dist;
            pos_Next = shared_next[k][j];  // according to theory, it should be [i][k], but [k][j] is what works
        }


        __syncthreads();

        shared_next[i][j] = pos_Next;
    }


    /* ---- Copy from shared memory to global memory ---- */

    __syncthreads();  // just to make sure that everyone is on the same page

    if (in_graph) {
        d_dist_pc[idx] = shared_dist[i][j];
        d_next_pc[idx] = shared_next[i][j];
    }
}



/*

The next two kernels calculate the tiles in the k-th row and column.

To calculate the COLUMN tiles, the equation is:

?? -- pos_dist = local_tile[i][k] + dep_tile[k][j]

In this case, i, the x-axis, is being held constant.


To calculate the ROW tiles, the equation is:

?? -- pos_dist = dep_tile[i][k] + local_tile[k][j]

In this case, j, the y-axis, is being held constant.

*/






// Blocked Step Two - COLUMN tiles
//


__global__ void blocked_Column(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                               int tile_k, int num_tiles_wide, int tile_width)
{

    /* ---- Find the 4th set of coordinates - the block's location in the TILE matrix ---- */

    // blockIdx.x / num_tiles_wide = which row it is on - the i value
    //
    int block_row = (blockIdx.x / num_tiles_wide);

    // blockIdx.x % num_tiles_wide = which column it is on - the j value
    //
    int block_col = (blockIdx.x % num_tiles_wide);


    /* ---- Escape Conditions ---- */

    // If our col # does not equal tile_k, then bail, as we aren't in the k-th column
    //
    if (block_col != tile_k) {
        return;
    }

    // Special case so as to avoid the dependent block
    //
    if ((block_col == tile_k) && (block_row == tile_k)) {
        return;
    }


//if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//    printf("tile_k = %d, blockIdx.x = %d, block_row = %d, block_col = %d\n\n", tile_k, blockIdx.x, block_row, block_col);
//}



    /* ---- Declare shared memory matrices ---- */

    // We need one for our block - the [k][j] value - and one for the dependent block,
    // which will provide the [i][k] value.

    __shared__ uint8_t this_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int this_next[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ uint8_t dep_dist[BLOCK_SIZE][BLOCK_SIZE];
    //__shared__ int dep_next[BLOCK_SIZE][BLOCK_SIZE];


    // Get our location data....

    /* ---- 1st set of coordinates ---- */

    // The thread's x/y coordinates for its position in the block, though expressed as i,j
    //
    int i = threadIdx.x;
    int j = threadIdx.y;


    /* ---- 2nd set of coordinates ---- */

    // We'll first find the memory location for our thread's location.
    //
    int tile_row_width = tile_width * num_tiles_wide;

    // To reach the row that our column block is on, we multiply block_row times
    // tile_row_width * tile_width.
    //
    int row_mem_start = block_row * (tile_row_width * tile_width);

    // We know that the column will be tile_k * tile_width over, so adding i to that
    // will give us our i position. For the j position, it will be the same regardless
    // of which TILE block we're in, so we'll multiply (j * tile_row_width) to get us
    // to the correct graph row and then add (tile_k * tile_width) + i to get the desired
    // (i,j) memory position - in that TILE row, that is.
    //
    int thread_mem_loc = (j * tile_row_width) + (tile_k * tile_width) + i;

    // this_idx = row_mem_start + thread_mem_loc - the start of the TILE row + the amount to move
    // forward in the graph itself to get to the correct graph (i,j) position.
    //
    int this_idx = row_mem_start + thread_mem_loc;


    // Now, we'll get the memory location for the thread's 'location' in the dependent block
    //
// update
    // To reach the k-th dependent block, we multiply our row_width by tile_k.
    //
    row_mem_start = tile_k * (tile_row_width * tile_width);

    // At this point, in memory, we are at the start of the k-th TILE row. Since j will
    // apply regardless of the i value, we multiply (j * num_vertices) to get how much
    // farther ahead we need to move to get to the correct graph row. For the i value,
    // since we know what the current k value is (tile_k) and know the tile_width, then
    // (tile_k * tile_width) will get us to the beginning of the desired block. Adding
    // i to that will finish the process.
    //
    thread_mem_loc = (j * tile_row_width) + (tile_k * tile_width) + i;

    // idx = row_mem_start + thread_mem_loc - the start of the TILE row + the amount to move
    // forward in the graph itself to get to the correct graph (i,j) position.
    //
    int dep_idx = row_mem_start + thread_mem_loc;


    /* ---- 3rd set of coordinates ---- */

    // These coordinates are the thread's (i,j) coordinates in the entire graph matrix AND
    // the coordinates of the thread's 'location' in the dependent block.

    // For OUR tile...

    // To determine the graph_j coordinate, we multiply block_row by tile_width and then
    // add j.
    //
    int graph_j = (block_row * tile_width) + j;

    // To determine the graph_i coordinate, we multiply block_col by tile_width and then
    // add i.
    //
    int graph_i = (block_col * tile_width) + i;


    // For the dependent tile...

    // To determine the graph_j coordinate, we multiply tile_k by tile_width and then
    // add j.
    //
    int dep_graph_j = (tile_k * tile_width) + j;

    // To determine the graph_i coordinate, we multiply tile_k by tile_width and then
    // add i.
    //
    int dep_graph_i = (tile_k * tile_width) + i;


    /* ---- Load our data into the shared memory matrixes ---- */

    // Since it's possible that the graph will not fully occupy the block, then we need to
    // determine whether or not this thread is actually representing an element in the graph.
    // If it is not, we'll set the dist and next values such that it won't affect the proper
    // processing of the graph.

    // Determine whether or not the thread is in the actual graph or just padding.
    //
    bool this_in_graph = false;

    if ((graph_i < num_vertices) && (graph_j < num_vertices)) {
        this_in_graph = true;
    }


    // For THIS tile...

    // Load the thread's dist and next values into its shared memory location.
    //
    if (this_in_graph) {
        this_dist[i][j] = d_dist_pc[this_idx];
        this_next[i][j] = d_next_pc[this_idx];
    } else {
        this_dist[i][j] = INF;
        this_next[i][j] = -1;
    }

    // Stop and wait until everyone has loaded their data into the shared matrices
    //
    __syncthreads();


    // Now, we need to load the data from our thread's 'position' in the dep tile
    // into its shared memory matrices.

    // Determine whether or not the thread is in the actual graph or just padding.
    //
    bool dep_in_graph = false;

    if ((dep_graph_i < num_vertices) && (dep_graph_j < num_vertices)) {
        dep_in_graph = true;
    }

    // For the dep tile...

    // Load the thread's dist and next values into its shared memory location.
    //
    if (dep_in_graph) {
        dep_dist[i][j] = d_dist_pc[dep_idx];
        //dep_next[i][j] = d_next_pc[dep_idx];
    } else {
        dep_dist[i][j] = INF;
        //dep_next[i][j] = -1;
    }

    // Stop and wait until everyone has loaded their data into the shared dep matrices
    //
    __syncthreads();


    /* ---- Now to run the F-W algorithm ---- */

    // As we are processing the COLUMN, then the equation will be:
    //
    // pos_dist = dep_dist[i][k] + this_dist[k][j]

    // Temp variables for possible new values, initialized to starting values
    //
    int pos_Dist = this_dist[i][j];
    int pos_Next = this_next[i][j];

    // The loop
    //
    for (int k=0; k < BLOCK_SIZE; k++)
    {

        //pos_Dist = 2; 
        //pos_Dist = tile_k + 1; 

        //pos_Dist = this_dist[i][j]; 
        //pos_Dist = dep_dist[i][j]; 


        //this_dist[i][j] = pos_Dist;


        pos_Dist = dep_dist[i][k] + this_dist[k][j]; 

        __syncthreads();

        if (pos_Dist < this_dist[i][j])
        {
            this_dist[i][j] = pos_Dist;
            pos_Next = this_next[k][j];  // according to theory, it should be [i][k], but [k][j] is what works
        }


        __syncthreads();

        this_next[i][j] = pos_Next;
    }


    /* ---- Copy from shared memory to global memory ---- */

    __syncthreads();  // just to make sure that everyone is on the same page

    if (this_in_graph) {
        d_dist_pc[this_idx] = this_dist[i][j];
        d_next_pc[this_idx] = this_next[i][j];
    }
}


__global__ void blocked_Row(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                            int tile_k, int num_tiles_wide, int tile_width)
{

    /* ---- Find the 4th set of coordinates - the block's location in the TILE matrix ---- */

    // blockIdx.x / num_tiles_wide = which row it is on - the i value
    //
    int block_row = (blockIdx.x / num_tiles_wide);

    // blockIdx.x % num_tiles_wide = which column it is on - the j value
    //
    int block_col = (blockIdx.x % num_tiles_wide);


    /* ---- Escape Conditions ---- */

    // If our row # does not equal tile_k, then bail, as we aren't in the k-th row
    //
    if (block_row != tile_k) {
        return;
    }

    // Special case so as to avoid the dependent block
    //
    if ((block_col == tile_k) && (block_row == tile_k)) {
        return;
    }


//if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//    printf("tile_k = %d, blockIdx.x = %d, block_row = %d, block_col = %d\n\n", tile_k, blockIdx.x, block_row, block_col);
//}


    /* ---- Declare shared memory matrices ---- */

    // We need one for our block - the [i][k] value - and one for the dependent block,
    // which will provide the [k][j] value.

    __shared__ uint8_t this_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int this_next[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ uint8_t dep_dist[BLOCK_SIZE][BLOCK_SIZE];
    //__shared__ int dep_next[BLOCK_SIZE][BLOCK_SIZE];


    // Get our location data....

    /* ---- 1st set of coordinates ---- */

    // The thread's x/y coordinates for its position in the block, though expressed as i,j
    //
    int i = threadIdx.x;
    int j = threadIdx.y;


    /* ---- 2nd set of coordinates ---- */

    // We'll first find the memory location for our thread's location.
    //
    int tile_row_width = tile_width * num_tiles_wide;

    // To reach the row that our block is on, we multiply block_row times
    // tile_row_width * tile_width.
    //
    int row_mem_start = block_row * (tile_row_width * tile_width);

    // We know that the block will be block_col * tile_width over, so adding i to that
    // will give us our i position. For the j position, it will be the same regardless
    // of which TILE block we're in, so we'll multiply (j * tile_row_width) to get us
    // to the correct graph row and then add (tile_k * tile_width) + i to get the desired
    // (i,j) memory position - in that TILE row, that is.
    //
    int thread_mem_loc = (j * tile_row_width) + (block_col * tile_width) + i;

    // this_idx = row_mem_start + thread_mem_loc - the start of the TILE row + the amount to move
    // forward in the graph itself to get to the correct graph (i,j) position.
    //
    int this_idx = row_mem_start + thread_mem_loc;


    // Now, we'll get the memory location for the thread's 'location' in the dependent block
    //
// update
    // To reach the k-th dependent block, we multiply our row_width by tile_k.
    //
    row_mem_start = tile_k * (tile_row_width * tile_width);

    // At this point, in memory, we are at the start of the k-th TILE row. Since j will
    // apply regardless of the i value, we multiply (j * num_vertices) to get how much
    // farther ahead we need to move to get to the correct graph row. For the i value,
    // since we know what the current k value is (tile_k) and know the tile_width, then
    // (tile_k * tile_width) will get us to the beginning of the desired block. Adding
    // i to that will finish the process.
    //
    thread_mem_loc = (j * tile_row_width) + (tile_k * tile_width) + i;

    // idx = row_mem_start + thread_mem_loc - the start of the TILE row + the amount to move
    // forward in the graph itself to get to the correct graph (i,j) position.
    //
    int dep_idx = row_mem_start + thread_mem_loc;


    /* ---- 3rd set of coordinates ---- */

    // These coordinates are the thread's (i,j) coordinates in the entire graph matrix AND
    // the coordinates of the thread's 'location' in the dependent block.

    // For OUR tile...

    // To determine the graph_j coordinate, we multiply block_row by tile_width and then
    // add j.
    //
    int graph_j = (block_row * tile_width) + j;

    // To determine the graph_i coordinate, we multiply block_col by tile_width and then
    // add i.
    //
    int graph_i = (block_col * tile_width) + i;


    // For the dependent tile...

    // To determine the graph_j coordinate, we multiply tile_k by tile_width and then
    // add j.
    //
    int dep_graph_j = (tile_k * tile_width) + j;

    // To determine the graph_i coordinate, we multiply tile_k by tile_width and then
    // add i.
    //
    int dep_graph_i = (tile_k * tile_width) + i;


    /* ---- Load our data into the shared memory matrixes ---- */

    // Since it's possible that the graph will not fully occupy the block, then we need to
    // determine whether or not this thread is actually representing an element in the graph.
    // If it is not, we'll set the dist and next values such that it won't affect the proper
    // processing of the graph.

    // Determine whether or not the thread is in the actual graph or just padding.
    //
    bool this_in_graph = false;

    if ((graph_i < num_vertices) && (graph_j < num_vertices)) {
        this_in_graph = true;
    }


    // For THIS tile...

    // Load the thread's dist and next values into its shared memory location.
    //
    if (this_in_graph) {
        this_dist[i][j] = d_dist_pc[this_idx];
        this_next[i][j] = d_next_pc[this_idx];
    } else {
        this_dist[i][j] = INF;
        this_next[i][j] = -1;
    }

    // Stop and wait until everyone has loaded their data into the shared matrices
    //
    __syncthreads();


    // Now, we need to load the data from our thread's 'position' in the dep tile
    // into its shared memory matrices.

    // Determine whether or not the thread is in the actual graph or just padding.
    //
    bool dep_in_graph = false;

    if ((dep_graph_i < num_vertices) && (dep_graph_j < num_vertices)) {
        dep_in_graph = true;
    }

    // For the dep tile...

    // Load the thread's dist and next values into its shared memory location.
    //
    if (dep_in_graph) {
        dep_dist[i][j] = d_dist_pc[dep_idx];
        //dep_next[i][j] = d_next_pc[dep_idx];
    } else {
        dep_dist[i][j] = INF;
        //dep_next[i][j] = -1;
    }

    // Stop and wait until everyone has loaded their data into the shared dep matrices
    //
    __syncthreads();


    /* ---- Now to run the F-W algorithm ---- */

    // As we are processing the ROW, then the equation will be:
    //
    // pos_dist = this_dist[i][k] + dep_dist[k][j]

    // Temp variables for possible new values, initialized to starting values
    //
    int pos_Dist = this_dist[i][j];
    int pos_Next = this_next[i][j];

    // The loop
    //
    for (int k=0; k < BLOCK_SIZE; k++)
    {

        //pos_Dist = 1; 
        //pos_Dist = tile_k + 1; 
        //pos_Dist = dep_dist[i][j]; 

        //this_dist[i][j] = pos_Dist;


        pos_Dist = this_dist[i][k] + dep_dist[k][j]; 

        __syncthreads();

        if (pos_Dist < this_dist[i][j])
        {
            this_dist[i][j] = pos_Dist;
            pos_Next = this_next[k][j];  // according to theory, it should be [i][k], but [k][j] is what works
        }


        __syncthreads();

        this_next[i][j] = pos_Next;
    }


    /* ---- Copy from shared memory to global memory ---- */

    __syncthreads();  // just to make sure that everyone is on the same page

    if (this_in_graph) {
        d_dist_pc[this_idx] = this_dist[i][j];
        d_next_pc[this_idx] = this_next[i][j];
    }
}


// For Step Four, we process all other blocks
//
__global__ void blocked_Independent(uint8_t* d_dist_pc, int* d_next_pc, int num_vertices, int num_rows,
                                    int tile_k, int num_tiles_wide, int tile_width)
{
    /*

        Note: For this stage, while we are comparing the calculated value against whatever
              is in block(i,j), the numbers that we are using for that calculation come from
              blocks in the semi-dependent row and column blocks. Thus, our shared memory
              arrays will be for the semi-dependent blocks, not the block that this thread is in.

    */


    /* ---- Find the 4th set of coordinates - the block's location in the TILE matrix ---- */

    // blockIdx.x / num_tiles_wide = which row it is on - the i value
    //
    int block_row = (blockIdx.x / num_tiles_wide);

    // blockIdx.x % num_tiles_wide = which column it is on - the j value
    //
    int block_col = (blockIdx.x % num_tiles_wide);


    /* ---- Escape Conditions ---- */

    // If our col # or row # equals tile_k, then bail, as we DON'T want to be in the k-th column or row
    //
    if ((block_col == tile_k) || (block_row == tile_k)) {
        return;
    }

    // Special case so as to avoid the dependent block
    //
    if ((block_col == tile_k) && (block_row == tile_k)) {
        return;
    }


//if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//    printf("tile_k = %d, blockIdx.x = %d, block_row = %d, block_col = %d\n\n", tile_k, blockIdx.x, block_row, block_col);
//}


    /* ---- Declare shared memory matrices ---- */

    // We need one for the row block - the [i][k] value - one for the column
    // block - the [k][j] value - and one for the next matrix.

    __shared__ uint8_t row_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint8_t col_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int this_next[BLOCK_SIZE][BLOCK_SIZE];


    // Get our location data....

    /* ---- 1st set of coordinates ---- */

    // The thread's x/y coordinates for its position in the block, though expressed as i,j
    //
    int i = threadIdx.x;
    int j = threadIdx.y;


    /* ---- 2nd set of coordinates ---- */

    // We'll first find the memory location for our thread's location.
    //
    int tile_row_width = tile_width * num_tiles_wide;

    // To reach the row that our column block is on, we multiply block_row times
    // tile_row_width * tile_width.
    //
    int row_mem_start = block_row * (tile_row_width * tile_width);

//update
    // We know that the column will be tile_k * tile_width over, so adding i to that
    // will give us our i position. For the j position, it will be the same regardless
    // of which TILE block we're in, so we'll multiply (j * tile_row_width) to get us
    // to the correct graph row and then add (tile_k * tile_width) + i to get the desired
    // (i,j) memory position - in that TILE row, that is.
    //
    int thread_mem_loc = (j * tile_row_width) + (block_col * tile_width) + i;

    // idx = row_mem_start + thread_mem_loc - the start of the TILE row + the amount to move
    // forward in the graph itself to get to the correct graph (i,j) position.
    //
    int idx = row_mem_start + thread_mem_loc;


    // Now to get the index for the col block
    //
//update
    // First, we multiply our row - block_row - by (tile_row_width * tile_width).

    row_mem_start = block_row * (tile_row_width * tile_width);

//update
    // At this point, in memory, we are at the start of the k-th TILE row. Since j will
    // apply regardless of the i value, we multiply (j * num_vertices) to get how much
    // farther ahead we need to move to get to the correct graph row. For the i value,
    // since we know what the current k value is (tile_k) and know the tile_width, then
    // (tile_k * tile_width) will get us to the beginning of the desired block. Adding
    // i to that will finish the process.
    //
    thread_mem_loc = (j * tile_row_width) + (tile_k * tile_width) + i;

    // idx = row_mem_start + thread_mem_loc - the start of the TILE row + the amount to move
    // forward in the graph itself to get to the correct graph (i,j) position.
    //
    int col_idx = row_mem_start + thread_mem_loc;


    // Now to get the index for the row block
    //
//update
    // First, we multiply our row - block_row - by (tile_row_width * tile_width).

    row_mem_start = tile_k * (tile_row_width * tile_width);

//update
    // At this point, in memory, we are at the start of the k-th TILE row. Since j will
    // apply regardless of the i value, we multiply (j * num_vertices) to get how much
    // farther ahead we need to move to get to the correct graph row. For the i value,
    // since we know what the current k value is (tile_k) and know the tile_width, then
    // (tile_k * tile_width) will get us to the beginning of the desired block. Adding
    // i to that will finish the process.
    //
    thread_mem_loc = (j * tile_row_width) + (block_col * tile_width) + i;

    // idx = row_mem_start + thread_mem_loc - the start of the TILE row + the amount to move
    // forward in the graph itself to get to the correct graph (i,j) position.
    //
    int row_idx = row_mem_start + thread_mem_loc;


    /* ---- 3rd set of coordinates ---- */

    // These coordinates are the thread's (i,j) coordinates in the entire graph matrix AND
    // the coordinates of the thread's 'location' in the semi-dependent blocks.

    // For OUR tile...

    // To determine the graph_j coordinate, we multiply block_row by tile_width and then
    // add j.
    //
    int graph_j = (block_row * tile_width) + j;

    // To determine the graph_i coordinate, we multiply block_col by tile_width and then
    // add i.
    //
    int graph_i = (block_col * tile_width) + i;


    // For the column tile...

//update all below PRN
    // To determine the graph_j coordinate, we multiply tile_k by tile_width and then
    // add j.
    //
    int col_graph_j = (block_row * tile_width) + j;

    // To determine the graph_i coordinate, we multiply tile_k by tile_width and then
    // add i.
    //
    int col_graph_i = (tile_k * tile_width) + i;


    // For the row tile...

    // To determine the graph_j coordinate, we multiply tile_k by tile_width and then
    // add j.
    //
    int row_graph_j = (tile_k * tile_width) + j;

    // To determine the graph_i coordinate, we multiply tile_k by tile_width and then
    // add i.
    //
    int row_graph_i = (block_col * tile_width) + i;


    /* ---- Load our data into the shared memory matrixes ---- */

    // Since it's possible that the graph will not fully occupy the block, then we need to
    // determine whether or not this thread is actually representing an element in the graph.
    // If it is not, we'll set the dist and next values such that it won't affect the proper
    // processing of the graph.

    // Determine whether or not the thread is in the actual graph or just padding.
    //
    bool this_in_graph = false;

    if ((graph_i < num_vertices) && (graph_j < num_vertices)) {
        this_in_graph = true;
    }

    // Determine whether or not the col thread is in the actual graph or just padding.
    //
    bool col_in_graph = false;

    if ((col_graph_i < num_vertices) && (col_graph_j < num_vertices)) {
        col_in_graph = true;
    }

    // Determine whether or not the row thread is in the actual graph or just padding.
    //
    bool row_in_graph = false;

    if ((row_graph_i < num_vertices) && (row_graph_j < num_vertices)) {
        row_in_graph = true;
    }


    // Load the thread's next value into its shared memory location.
    //
    if (this_in_graph) {
        this_next[i][j] = d_next_pc[idx];
    } else {
        this_next[i][j] = -1;
    }

    // Load the col thread's next value into its shared memory location.
    //
    if (col_in_graph) {
        col_dist[i][j] = d_dist_pc[col_idx];
    } else {
        col_dist[i][j] = INF;
    }

    // Load the row thread's next value into its shared memory location.
    //
    if (row_in_graph) {
        row_dist[i][j] = d_dist_pc[row_idx];
    } else {
        row_dist[i][j] = INF;
    }

    // Stop and wait until everyone has loaded their data into the shared matrices
    //
    __syncthreads();


    /* ---- Grab our dist and next values and store them in temp variables ---- */

    int current_dist = d_dist_pc[idx];
    int current_next = d_next_pc[idx];

    int pos_Dist = 0;


    // Stop and wait until everyone has loaded their data into the shared matrices
    // and has loaded their current values into temp variables.
    //
    __syncthreads();


    /* ---- Now to run the F-W algorithm ---- */

    // The loop
    //
    for (int k=0; k < BLOCK_SIZE; k++)
    {
        //pos_Dist = tile_k + 2; 
        //shared_dist[i][j] = pos_Dist;

        //current_dist = tile_k + 3;

        //current_dist = current_dist;

        //current_dist = col_dist[i][j]; 
        //current_dist = row_dist[i][j]; 

        //current_dist = row_dist[i][k] + col_dist[k][j]; 


        pos_Dist = row_dist[i][k] + col_dist[k][j]; 

        __syncthreads();

        if (pos_Dist < current_dist)
        {
            current_dist = pos_Dist;
            current_next = this_next[k][j];  // according to theory, it should be [i][k], but [k][j] is what works
        }

        __syncthreads();
        
        this_next[i][j] = current_next;

    }


    /* ---- Copy from shared memory to global memory ---- */

    __syncthreads();  // just to make sure that everyone is on the same page

    if (this_in_graph) {
        d_dist_pc[idx] = current_dist;
        d_next_pc[idx] = this_next[i][j];
    }
}


