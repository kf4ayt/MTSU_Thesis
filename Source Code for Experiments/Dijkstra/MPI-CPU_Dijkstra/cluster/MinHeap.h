

/*****************************************************************

The code for this binary heap implementation was downloaded from
https://www.geeksforgeeks.org/binary-heap/. All credit goes to
the author(s) of the article.

Site last accessed: April 28, 2021

*****************************************************************/


#include <iostream>
#include <climits>

using namespace std;

// heap node

struct heapNode
{
    int vertex;
    int dist;
};


// Prototype of a utility function to swap two integers 
void swap(heapNode *x, heapNode *y); 


// A class for a min binary heap of integers 
//
class MinHeap 
{ 
    // These are private
    //
    heapNode* heap_array;    // array that stores the heap
    int array_size;     // size of heap_array
    int current_size;   // # of items currently in the heap

public: 
    // Constructor 
    MinHeap(int array_size); 
   
    // Destructor 
    ~MinHeap(); 
   
    // Returns the index of the parent of a given index
    int parent(int i);
    
    // Returns the index of the left child of the node
    // in the given index
    int left_child(int i);
    
    // Returns the index of the right child of the node
    // in the given index
    int right_child(int i);
    
    // Returns the value of the root element, which, by definition,
    // will also be the element with the min value
    heapNode getMin();

    // Extracts the root element and restructures the
    // heap as necessary
    heapNode extractMin(); 
    
    // Inserts a new vertex (aka key) into the heap
    void insertKey(heapNode k); 

    // Changes the -value- in the heap_array whose
    // element # is given and restructures the heap
    // as necessary
    void decreaseKey(int i, int new_val);

    // Removes the key stored at the specified index and
    // restructures the heap as necessary
    void deleteKey(int i); 
 
    // 'Heapifies' the subtree whose root index is the
    // location in the heap_array given.
    void heapify(int i); 

    // Returns heap size
    int heapSize();

    // Returns if the given vertex is in the heap
    bool isPresent(int vertex);
}; 


