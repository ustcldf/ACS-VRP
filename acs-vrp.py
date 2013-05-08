#!/usr/bin/python2.7

import math
import random
import psycopg2


def nnt(graph,startNode):
  # Compute the nearest-neighbour-tour.
  #
  # graph: 2D array with numberOfNodes rows and columns and the weight of the edges as values.
  # startNode: the node where the tour begins and ends.
  # returns a list containing the nearest-neighbour-tour.

  tour = [startNode]
  remNodes = range(len(graph)) # remaining nodes
  remNodes.remove(startNode)
  curNode = startNode # current node
  for n in range(len(graph)-1):
    dist = [ (graph[curNode][i],i) for i in remNodes ] # list with length from curNode to i in the form (length, i)
    remNodes.remove(min(dist)[1]) # remove node that is nearest to curNode
    tour.append(min(dist)[1]) # append node that is nearest to curNode
    curNode = min(dist)[1] # set node that is nearest to curNode as curNode
  tour.append(startNode) # append the starting node to end of tour (so we have a cycle)
  return tour


def gtl(graph,tour):
  # Get the length of a tour.
  #
  # graph: 2D array with numberOfNodes rows and columns and the weight of the edges as values.
  # tour: list containing the tour.
  # returns the length of the tour.

  length = 0
  for i in range(len(tour)-1):
    length += graph[tour[i]][tour[i+1]] # add up nodes
  return length


def checkForBestTour(graph, tours, oldBestTour):
  # Check for new best tour.
  #
  # graph: 2D array with numberOfNodes rows and columns and the weight of the edges as values.
  # tours: list of lists (list of tours).
  # oldBestTour: list of nodes (the best tour so far).
  # returns a list containing the best tour (might be oldBestTour).

  best = float('inf')
  bestT = []

  for t in tours:
    length = gtl(graph, t)
    if (length < best):
      best = length
      bestT = t

  if (not oldBestTour): # bestTour is empty
    return bestT
  elif (best <= gtl(graph, oldBestTour)):
    return bestT

  return oldBestTour


def isEdgeOfBestTour(bestTour, r, s):
  # check if (r,s) is an edge of the best tour.
  #
  # bestTour: list containing the nodes of the best tour.
  # r: first node.
  # s: second node.
  # returns true if (r,s) is an edge of bestTour, false else.

  edges = []

  for i in range(len(bestTour)-1):
    edges.append((bestTour[i],bestTour[i+1]))

  return (r,s) in bestTour


def tau0(graph):
  # tau0 is (number of nodes * length of nearestNeighbourTour)^-1, default value for various formulas
  #
  # graph: 2D array with numberOfNodes rows and columns and the weight of the edges as values.
  # returns the value of tau0.

  return 1./(len(graph)*gtl(graph,nnt(graph,0)))


def haversine(lat0, lon0, lat1, lon1):
  # calculate the distance between two points on earth (inaccurate, no height differences?)
  #
  # lat0: latitude of the first point.
  # lon0: longitude of the first point.
  # lat1: latitude of the second point.
  # lon1: longitude of the second point.
  # returns distance between the two points in km.

  R = 6371 # radius earth
  dLat = lat1-lat0
  dLon = lon1-lon0
  a = math.sin(dLat/2)**2 + math.sin(dLon/2)**2 * math.cos(lat0) * math.cos(lat1)
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  return (R*c)


def getDistance(a,b):
  # Calculate the distance between two openstreetmap node-indices.
  #
  # a: first osm index.
  # b: second osm index.
  # returns the distance between a and b (unit?)

  print 'From: ', a
  print 'To: ', b
  print
  if (a == b):
    return 0
  con = psycopg2.connect(database='osmpgrouting', user='postgres')
  cur = con.cursor()
  query = "SELECT * FROM shortest_path('SELECT gid as id, source::integer, target::integer, length::double precision as cost FROM ways'," + str(a) + "," + str(b) + ", false, false);"
  cur.execute(query)
  rows = cur.fetchall()
  s = 0
  for row in rows:
    s = s + row[2]
  return s


def probfunc(l):
  # determine an index, where each index has the probability of being picked of the value of the
  # array at that index.
  #
  # l: list, where sum(l) = 1, e.g. [0.5, 0.2, 0.1, 0.2].
  # returns an index based on probability.

  r = random.random() # get random number
  for i in range(0,len(l)):
    nl = [ l[j] for j in range(0,i+1) ] # create list, example for every loop: [0.1] then [0.1,0.5] then [0.1,0.5,0.1]...
    if (r < sum(nl)): # if the the random number is less than the sum, the position of that element gets returned
      return i
  print 'nothing found in probfunc' # list empty?
  print l
  print 'sum = ', sum(l)
  print
  if (sum(l) == 0): # items in list of probabilities all 0?
    print 'sum(l) = 0'
  for i in range(0,len(l)): # return the first non-zero element
    if (l[i] != 0):
      return i


def localUpdatingRule(pheromone, lastNode, currentNode, tau0val): 
  # update an edge using the local updating rule.
  #
  # pheromone: list containing pheromone values.
  # lastNode: the previous node of the ant.
  # currentNode: the node where the ant is now.

  rho = 0.1
  nval = (1-rho)*pheromone[lastNode][currentNode] + rho*tau0val
  pheromone[lastNode][currentNode] = nval # (lastNode,currentNode) != (currentNode,lastNode) ???


def globalUpdatingRule(graph, pheromone, bestTour):
  # Update the whole graph using the global updating rule.
  #
  # graph: 2D array with numberOfNodes rows and columns and the weight of the edges as values.
  # pheromone: 2D array with numberOfNodes rows and columns and the pheromoneamount of the edges as values.
  # bestTour: array containing the best tour so far

  for r in range(len(graph)):
    for s in range(len(graph)):
      alpha = 0.1
      nval = (1-alpha) * pheromone[r][s]
      if isEdgeOfBestTour(bestTour, r, s):
        nval += alpha * (1/gtl(graph, bestTour))
      pheromone[r][s] = nval # (r,s) != (s,r) ???


def stateTransitionRule(graph, pheromone, currentNode, remaining):
  # Choose the next node.
  #
  # graph: 2D array with numberOfNodes rows and columns and the weight of the edges as values.
  # pheromone: 2D array with numberOfNodes rows and columns and the pheromoneamount of the edges as values.
  # currentNode: the current position of the ant.
  # returns the next node.

  q0 = 0.9
  beta = 2
  q = random.random()

  if (q <= q0):
    maxval = 0
    for i in remaining:
      tmp = pheromone[currentNode][i] / (graph[currentNode][i] ** beta)
      if ( tmp >= maxval ):
        maxval = tmp
        s = i
  else:
    prob = [ 0 for k in range(len(graph)) ]
    sumval = 0
    for i in remaining:
      tmp = pheromone[currentNode][i] / (graph[currentNode][i] ** beta)
      sumval += tmp
      if (sumval == 0):
        print 'sumval == 0'

    for i in remaining:
      tmp = pheromone[currentNode][i] / (graph[currentNode][i] ** beta)
      prob[i] = (tmp / sumval)

    s = probfunc(prob)

  return s


def chooseNext(graph, pheromone, remaining, tours):
  # For every ant: choose next node
  #
  # graph: 2D array with numberOfNodes rows and columns and the weight of the edges as values.
  # pheromone: 2D array with numberOfNodes rows and columns and the pheromoneamount of the edges as values.
  # remaining: list of remaining nodes
  # tours: list of tours

  ant = 0 # current ant
  if (len(tours[ant]) == numNodes): # all nodes visited
    for a in tours:
      a.append(a[0])
      ant += 1
  else:
    for a in tours: # for every ant tour
      oldPos = a[len(a)-1]
      newPos = stateTransitionRule(graph, pheromone, oldPos, remaining[ant])
      localUpdatingRule(pheromone, oldPos, newPos, tau0(graph))
      a.append(newPos)
      remaining[ant].remove(newPos)
      ant += 1


def reset(remaining, tours, nodes, ants):
  # Reset remaining nodes and generated tours
  #
  # remaining: list containing the remaining nodes for every ant
  # tours: list containing the tour of every ant
  # nodes: list of nodes
  # ants: number of ants

  del remaining[:]
  for i in range(len(nodes)):
    remaining.append(nodes[:])

  del tours[:]
  for i in range(ants):
    tours.append([])


def positionAnts(ants, tours, numNodes, remaining):
  # Determine the start nodes of the ants.
  #
  # ants: number of ants.
  # tours: list of tours
  # numNodes: number of nodes.
  # remaining: unvisited nodes

  pos = range(numNodes)
  for ant in range(ants):
    p = random.choice(pos)
    tours[ant].append(p)
    remaining[ant].remove(p)
    pos.remove(p)


def addDepots(v, graph):
  # add v depots to the graph
  # vertex 0 gets replaced by those depots
  # distance between depots is inf

  l = []
  for g in graph: # create copy of graph
    l.append(g[:])
  l.pop(0) # remove adjacency list 0, the depots together form list 0
  for i in l:
    for j in range(v-1):
      i.insert(0,i[0]) # insert distance to depots 0,1,...
  for i in range(v):
    tmp = [float('inf') for j in range(v)] # distance between depots if inf
    for j in range(1,len(graph)):
      tmp.append(graph[0][j]) # append length from depot to customer
    l.insert(0, tmp)
  return l

def addDepots2(v, graph):
  l = [ [] for i in range(len(graph)+v-1) ]
  print l
  for i in len(graph):
    for j in len(graph[i]):
      if (i <= v):
        if (j <= v):
          None

if __name__ == '__main__':

  # osm indices (from table ways, column gid)
  #nodes = [1613619, 1570799, 1570804, 1570802, 1042694, 909169, 842014, 914263, 1209753, 1230487,
  #1313862, 42, 1337]

  # example non-osm-nodes
  nodes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

  # matrix with <numberOfNodes> rows and columns and osm-distances as values
  #graph = [ [ getDistance(i,j) for j in nodes ] for i in nodes ]

  # example non-osm-graph
  graph = [[0, 20, 14, 10, 2, 7, 3, 20, 3, 40, 1, 22, 6, 20],[20, 0, 2, 5, 4, 33, 10, 30, 3, 12, 42,
  19, 8, 21],[14, 2, 0, 10, 3, 22, 10, 3, 2, 33, 23, 7, 27, 5], [10, 5, 10, 0, 6, 20, 20, 11, 21, 21,
  73, 6, 14, 20],[2, 4, 3, 6, 0, 1, 2, 40, 12, 18, 17, 25, 30, 7], [7, 33, 22, 20, 1, 0, 40, 5, 3, 2,
  3, 11, 10, 33],[3, 10, 10, 20, 2, 40, 0, 8, 4, 7, 8, 24, 5, 13], [20, 30, 3, 11, 40, 5, 8, 0, 9, 11,
  4, 12, 3, 19],[3, 3, 2, 21, 12, 3, 4, 9, 0, 12, 42, 33, 21, 18], [40, 12, 33, 21, 18, 2, 7, 11, 12,
  0, 6, 3, 17, 4],[1, 42, 23, 73, 17, 3, 8, 4, 42, 6, 0, 6, 26, 8], [22, 19, 7, 6, 25, 11, 24, 12, 33,
  1, 6, 0, 20, 15],[6, 8, 27, 14, 30, 10, 5, 3, 21, 17, 26, 20, 0, 18],[20, 21, 5, 20, 7, 33, 13, 19,
  18, 4, 8, 15, 18, 0]]

  # number of nodes
  numNodes = len(graph)

  # initial pheromone amount
  #tau0val = tau0(graph)

  # pheromone array
  #pheromone = [ [ tau0val for i in range(numNodes) ] for j in range(numNodes) ]

  # number of ants
  #ants = 10

  # list of remaining nodes
  #remaining = [ nodes[:] for i in range(ants) ]

  # tours generated
  #tours = [ [] for i in range(ants) ]

  # the best tour
  #bestTour = []

  #positionAnts(ants, tours, numNodes, remaining)

  #for tmp in range(2):
  #  for count in range(1000):
  #    for i in range(numNodes):
  #      chooseNext(graph, pheromone, remaining, tours)
  #    bestTour = checkForBestTour(graph, tours, bestTour)
  #    globalUpdatingRule(graph, pheromone, bestTour)
  #    reset(remaining, tours, nodes, ants)
  #    positionAnts(ants, tours, numNodes, remaining)
  #  print 'best tour: ', bestTour
  #  print 'length of best tour: ', gtl(graph, bestTour)
  #  print 'length of nnt: ', gtl(graph, nnt(graph, 0))

  #graph = addDepots(numNodes-1, graph)
  graph = addDepots(1, graph)
  print 'nnt: ', nnt(graph, 0)
