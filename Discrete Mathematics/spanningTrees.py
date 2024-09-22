#!/usr/bin/python
# -*- coding: utf-8 -*-
from helpers import *

def About():
    print
    print("Task for Discrete Mathematics May 2018")
    print("~Anastasios Grammenos 9212 avgramme@ece.auth.gr~")
    print("Find all spanning trees from a graph")
    print

def Help():
    print("Help:")
    print("Use the function 'findSpanningTrees(Graph)' to find all spanning trees from a graph.")
    print("Note that the graph must be inserted as Incidence Matrix in a form of a list of lists.")
    print("For example:")

def findSpanningTrees(Graph=[]):
    About()
    if (Graph == []) :        
        Graph = [[1,0,0,1,1,0],
                 [1,1,0,0,0,1],
                 [0,1,1,0,1,0],
                 [0,0,1,1,0,1]]
        Help()

    printIncidenceMatrix(Graph)
    print
    print("All Spanning Trees:")
    Trees = []
    numOfTrees = findTrees(copyGraph(Graph) , [] , Trees)
    print
    print("Total = {0}".format(len(Trees)))
    print

    return Trees


if __name__ == "__main__" :
    Graph = []
    #Graph = [[2,1,1,1],[0,1,1,1]]
    #Graph = [[2,2,2]]
    findSpanningTrees(Graph)