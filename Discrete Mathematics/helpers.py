#!/usr/bin/python
# -*- coding: utf-8 -*-

# Αντιγράφει μια λίστα
def copyList(A):
    Copy = []
    for i in A:
        Copy.append(i)
    return Copy

# Αντιγράφει μια λίστα από λίστες (εδώ τον πίνακα πρόσπτωσης του γραφήματος)
def copyGraph(Graph):
    Copy = []
    for i in range(len(Graph)):
        Copy.append([])
        for j in range(len(Graph[i])):
            Copy[i].append(Graph[i][j])
    return Copy

# Κάνει συστολή μια ακμή του πίνακα πρόσπτωσης
def edgeContraction(Graph,edge):
    x , y = None , None
    for i in range(len(Graph)):
        if (Graph[i][edge] == 1) and (x == None):
            x = i
        elif (Graph[i][edge] == 1):
            y = i
    for i in range(len(Graph[0])):
        Graph[x][i] = Graph[x][i] + Graph[y][i]
    Graph[x][edge] = 0
    del Graph[y]
    Graph = edgeDeletion(Graph,edge)
    return Graph

# Διαγράφει μια ακμή του πίνακα πρόσπτωσης
def edgeDeletion(Graph,edge):
    for i in range(len(Graph)):
        Graph[i][edge] = 0
    return Graph

# Διαγράφει όλους τους βρόγχους του πίνακα πρόσπτωσης
def removeBronchus(Graph):   
    bronchusIndex = []
    for i in range(len(Graph)):
        for j in range(len(Graph[i])):
            if (Graph[i][j] == 2):
                bronchusIndex.append(j)
    for i in bronchusIndex:
        Graph = edgeDeletion(Graph,i)
    return Graph

# Ελέγχει εάν ο πίνακα πρόσπτωσης αναφαίρεται στο γράφημα με μία μόνο κορυφή και καμία ακμή
def checkSingleNode(Graph):
    if (len(Graph) == 1):
        for i in Graph[0]:
            if (i == 1):
                return False
        return True
    return False

# Ελέγχει από το πίνακα πρόσπτωσης εάν το γράφημα είναι συνδεδεμένο και εάν ναι επιστρέφει μία τυχαία ακμή
def checkConnected(Graph):
    edge = None
    for i in range(len(Graph)):
        degree = 0
        for j in range(len(Graph[i])):
            if (Graph[i][j] != 0):
                edge = j
            degree += Graph[i][j]
        if (degree == 0) :
            return False , None
    return True , edge

# Αναδρομική συνάρτηση που βρίσκει και αποθηκεύει όλα τα γεννητορικά δένδρα του γραφήματος και επιστρέφει τον αριθμό αυτών
def findTrees(Graph , memory , Trees):
    if (checkSingleNode(Graph)):
        Trees.append(memory)
        print('{0}: {1}'.format(len(Trees) , Trees[-1]))
        return 1

    Graph = removeBronchus(copyGraph(Graph))
    connected , edge = checkConnected(Graph)

    if not(connected):
        return 0

    deletion = findTrees(edgeDeletion(copyGraph(Graph),edge) , copyList(memory) , Trees)
    memory.append(edge)
    contraction = findTrees(edgeContraction(copyGraph(Graph),edge) , copyList(memory) , Trees)

    return deletion+contraction

# Εκτυπώνει το πίνακα πρόσπτωσης
def printIncidenceMatrix(Graph):
    print
    print("Incidence Matrix:")
    print('V\E: {0}'.format(range(len(Graph[0]))))
    for i in range(len(Graph)):
        print('v{0} : {1}'.format(i , Graph[i]))
