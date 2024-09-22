#!/usr/bin/python
# -*- coding: utf-8 -*-
import random

EdgeRandomThreshold = 0.5   # Probability of creating a edge between two nodes

def createMap(n):
    map = {}
    invMap= {}
    for i in range(0,n):
        while (len(map) < i+1):
            temp = random.randint(0,n-1)
            flag = True
            for j in range(len(map)):
                if map[j] == temp:
                    flag = False
                    break
            if flag:
                map.update({i:temp})
                invMap.update({temp:i})
    return invMap

# Input: Graph(list of lists)
# Output: IsomGraph(list of lists) , Map(dictionary with corresponding nodes INVERTED!)
def createIsomorphismGraph(Graph):
    numbOfNodes = len(Graph)
    map = createMap(numbOfNodes)
    newGraph = []
    for i in range(numbOfNodes):
        tempNode = []
        for j in range(numbOfNodes):
            if Graph[map[i]][map[j]] == 1:
                tempNode.append(1)
            else:
                tempNode.append(0)
        newGraph.append(tempNode)
    return newGraph , map

# Input: Number of nodes of the graph(int)
# Output: random Graph (list of lists)
def createRandomGraph(numbOfNodes):
    Graph = []
    for i in range(numbOfNodes):
        tempNode = []
        for j in range(numbOfNodes):
            #if j == i:
            #    tempNode.append(0)
            if j >= i:
                if random.random() >= EdgeRandomThreshold:
                    tempNode.append(1)
                else:
                    tempNode.append(0)
            else:
                tempNode.append(Graph[j][i])
        Graph.append(tempNode)
    return Graph