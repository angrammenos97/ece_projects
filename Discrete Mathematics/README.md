# Spanning Trees Finder

## Overview

This project aims to find all spanning trees of a graph using its incidence matrix, designed to help with graph theory concepts and spanning tree computations for the *"Discrete Mathematics"* subject.

## Getting Started

### Usage

1. **Library File**: `helpers.py`
   - Contains functions for manipulating incidence matrices and finding spanning trees.
   - Key functions:
     - `copyList(A)`: Creates a shallow copy of a list.
     - `copyGraph(Graph)`: Creates a deep copy of an incidence matrix.
     - `edgeContraction(Graph, edge)`: Contracts an edge in the incidence matrix.
     - `edgeDeletion(Graph, edge)`: Deletes an edge from the incidence matrix.
     - `removeBronchus(Graph)`: Removes loops (self-edges) from the incidence matrix.
     - `checkSingleNode(Graph)`: Checks if the graph has only one isolated node.
     - `checkConnected(Graph)`: Checks if the graph is connected and returns a random edge.
     - `findTrees(Graph, memory, Trees)`: Finds and stores all spanning trees of the graph recursively.
     - `printIncidenceMatrix(Graph)`: Prints the incidence matrix.

2. **Main Script**: `main.py`
   - **Functions**:
     - `About()`: Prints information about the project.
     - `Help()`: Provides usage instructions for the `findSpanningTrees` function.
     - `findSpanningTrees(Graph=[])`: Finds and prints all spanning trees of the graph. The graph is provided as an incidence matrix.

3. **Example Usage**:
    ```python
    if __name__ == "__main__":
        Graph = []
        # Uncomment and set a custom graph as needed
        # Graph = [[2,1,1,1],[0,1,1,1]]
        # Graph = [[2,2,2]]
        findSpanningTrees(Graph)
    ```

### Example Graph

The default graph used in the `findSpanningTrees` function is:
```python
Graph = [[1,0,0,1,1,0],
         [1,1,0,0,0,1],
         [0,1,1,0,1,0],
         [0,0,1,1,0,1]]
