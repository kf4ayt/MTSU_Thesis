

/*****************************************************************

The code for this binary heap implementation was downloaded from
https://www.geeksforgeeks.org/binary-heap/. All credit goes to
the author(s) of the article.

Site last accessed: April 28, 2021

*****************************************************************/


#include "MinHeap.h"

#include <iostream>
#include <climits>

using namespace std;


/* ---- Class function definitions ---- */

// Constructor: Builds a heap from a given array a[] of given size 
MinHeap::MinHeap(int size) 
{ 
    current_size = 0; 
    array_size = size; 
    heap_array = new heapNode[array_size];
} 


// Destructor: frees the memory allocated in the constructor 
MinHeap::~MinHeap() 
{ 
    delete[] heap_array;
}


// Returns the index of the parent of a given index
int MinHeap::parent(int i)
{
    return ((i-1) / 2);
}


// Returns the index of the left child of the node
// in the given index
int MinHeap::left_child(int i)
{
    return ((2*i) + 1);
}


// Returns the index of the right child of the node
// in the given index
int MinHeap::right_child(int i)
{
    return ((2*i) + 2);
}


// Returns the vertex/dist of the root element, which, by definition,
// will also be the element with the min value
heapNode MinHeap::getMin()
{
    if (current_size == 0) {
        heapNode temp;
        temp.vertex = -1;
        temp.dist = -1;

        heap_array[0] = temp;
    }

    return heap_array[0];
}


// Extracts the root element and restructures the
// heap as necessary 
//
heapNode MinHeap::extractMin()
{ 
    // if the size of the heap is 1, return the root
    if (current_size == 1)
    {
        current_size--;
        return heap_array[0];
    }

    // Store the min value (key aka vertex) in a temp variable,
    // put the key at the bottom of the heap in the root position,
    // reduce the heap's size by 1, and then, starting with the root,
    // Heapify things, so as to put the heap back into order.

    heapNode root = heap_array[0]; 
    heap_array[0] = heap_array[current_size-1]; 
    current_size--; 
    heapify(0); 

    return root; 
} 


// Inserts a new heapNode (vertex/dist) (aka key) into the heap
void MinHeap::insertKey(heapNode key) 
{ 
    // Sanity check - if the heap is at capacity, print
    // an error message and bail

    if (current_size == array_size)
    {
        cout << "Heap is full" << endl;
        return;
    }

    // We start by inserting the key at the end of the heap
    // and then do the swap thing until it is in its proper
    // place.

    current_size++;
    int i = current_size - 1;   // this will be the index after the last index in use
    heap_array[i] = key;        // store the value there

    // Now, we swap up PRN
    while ((i != 0) && (heap_array[parent(i)].dist > heap_array[i].dist))
    {
        swap(&heap_array[i], &heap_array[parent(i)]);
        i = parent(i);
    }
} 


// Changes the dist value for the vertex passed
//
void MinHeap::decreaseKey(int vertex, int new_dist) 
{ 
    // first, find the index with the desired vertex
    int idx = 0;

    for (int a=0; a<current_size; a++)
    {
        if (heap_array[a].vertex == vertex)
        {
            idx = a;
            break;
        }
    }

    // now make the assignment

    heap_array[idx].dist = new_dist; 

    // while the parent is bigger than the child (i)
    // swap the values and move up one level. If
    // the parent is the root node, stop.

    heapify(parent(idx));
} 


// Removes the key stored at the specified index and
// restructures the heap as necessary
void MinHeap::deleteKey(int i) 
{ 
    decreaseKey(i, INT_MIN);    // decrease value of key to neg-INF
    extractMin();               // removes the key from the heap (it will be the min value present)
} 


// 'Heapifies' the subtree whose root index is the
// location in the heap_array given.
void MinHeap::heapify(int i) 
{ 
    // get the indices for the children
    int left = left_child(i);
    int right = right_child(i);

    // Since this is a min heap, that means that the smallest -value-
    // will be at the index given - i
    int smallest = i; 

    /* ---- These are our base cases ---- */

    if ((left < current_size)   // the index isn't higher than the highest index being used
        &&
        (heap_array[left].dist < heap_array[i].dist))     // it's smaller than the LEFT child
    {
        smallest = left;
    }

    if ((right < current_size)   // the index isn't higher than the highest index being used
        &&
        (heap_array[right].dist < heap_array[i].dist))     // it's smaller than the RIGHT child
    {
        smallest = right;
    }

    /* ---- If it doesn't meet a base case... ---- */

    if (smallest != i)
    {
        swap(&heap_array[i], &heap_array[smallest]);     // swap the values
        heapify(smallest);                           // the recursive call
    }

    /* ---- If it doesn't meet any of the above 3 ifs, then we're in serious trouble... ---- */
} 


/* ---- 'Utility' functions ---- */

// Returns size of current heap
//
int MinHeap::heapSize()
{
    return current_size;
}


// Returns if the given vertex is in the heap
//
bool MinHeap::isPresent(int vertex) 
{ 
    for (int a=0; a<current_size; a++)
    {
        if (heap_array[a].vertex == vertex)
        {
            return true;
        }
    }

    return false;
}


// Swaps two elements
//
void swap(heapNode *x, heapNode *y) 
{ 
    heapNode temp = *x; 
    *x = *y; 
    *y = temp; 
} 


