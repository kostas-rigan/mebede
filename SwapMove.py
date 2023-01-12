from VRP_Model import *

class SwapMove(object):

    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = None

    def Initialize(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = 10 ** 9

    

def StoreBestSwapMove(self, firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex, moveCost, costChangeFirstRoute, costChangeSecondRoute, sm):
        sm.positionOfFirstRoute = firstRouteIndex
        sm.positionOfSecondRoute = secondRouteIndex
        sm.positionOfFirstNode = firstNodeIndex
        sm.positionOfSecondNode = secondNodeIndex
        sm.costChangeFirstRt = costChangeFirstRoute
        sm.costChangeSecondRt = costChangeSecondRoute
        sm.moveCost = moveCost

def ApplySwapMove(self, sm):
       oldCost = self.CalculateTotalCost(self.sol)
       rt1 = self.sol.routes[sm.positionOfFirstRoute]
       rt2 = self.sol.routes[sm.positionOfSecondRoute]
       b1 = rt1.sequenceOfNodes[sm.positionOfFirstNode]
       b2 = rt2.sequenceOfNodes[sm.positionOfSecondNode]
       rt1.sequenceOfNodes[sm.positionOfFirstNode] = b2
       rt2.sequenceOfNodes[sm.positionOfSecondNode] = b1

       if (rt1 == rt2):
           rt1.cost += sm.moveCost
       else:
           rt1.cost += sm.costChangeFirstRt
           rt2.cost += sm.costChangeSecondRt
           rt1.load = rt1.load - b1.demand + b2.demand
           rt2.load = rt2.load + b1.demand - b2.demand

       self.sol.cost += sm.moveCost

       newCost = self.CalculateTotalCost(self.sol)
       # debuggingOnly
       if abs((newCost - oldCost) - sm.moveCost) > 0.0001:
           print('Cost Issue')

def FindBestSwapMove(self, sm):
        for firstRouteIndex in range(0, len(self.sol.routes)):
            rt1:Route = self.sol.routes[firstRouteIndex]
            for secondRouteIndex in range (firstRouteIndex, len(self.sol.routes)):
                rt2:Route = self.sol.routes[secondRouteIndex]
                for firstNodeIndex in range (1, len(rt1.sequenceOfNodes) - 1):
                    startOfSecondNodeIndex = 1
                    if rt1 == rt2:
                        startOfSecondNodeIndex = firstNodeIndex + 1
                    for secondNodeIndex in range (startOfSecondNodeIndex, len(rt2.sequenceOfNodes) - 1):

                        a1 = rt1.sequenceOfNodes[firstNodeIndex - 1]
                        b1 = rt1.sequenceOfNodes[firstNodeIndex]
                        c1 = rt1.sequenceOfNodes[firstNodeIndex + 1]

                        a2 = rt2.sequenceOfNodes[secondNodeIndex - 1]
                        b2 = rt2.sequenceOfNodes[secondNodeIndex]
                        c2 = rt2.sequenceOfNodes[secondNodeIndex + 1]

                        moveCost = None
                        costChangeFirstRoute = None
                        costChangeSecondRoute = None

                        if rt1 == rt2:
                            if firstNodeIndex == secondNodeIndex - 1:
                                # case of consecutive nodes swap
                                costRemoved = self.distanceMatrix[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + self.distanceMatrix[b1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1) + \
                                              self.distanceMatrix[b2.ID][c2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-2)
                                costAdded = self.distanceMatrix[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + self.distanceMatrix[b2.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1) + \
                                            self.distanceMatrix[b1.ID][c2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-2)
                                moveCost = costAdded - costRemoved
                            else:

                                costRemoved1 = self.distanceMatrix[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + self.distanceMatrix[b1.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                                costAdded1 = self.distanceMatrix[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + self.distanceMatrix[b2.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                                costRemoved2 = self.distanceMatrix[a2.ID][b2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + self.distanceMatrix[b2.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                                costAdded2 = self.distanceMatrix[a2.ID][b1.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + self.distanceMatrix[b1.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)

                                moveCost = costAdded1 + costAdded2 - (costRemoved1 + costRemoved2)
                        else:
                            if rt1.load - b1.demand + b2.demand > self.capacity:
                                continue
                            if rt2.load - b2.demand + b1.demand > self.capacity:
                                continue

                            costRemoved1 = self.distanceMatrix[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + self.distanceMatrix[b1.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                            costAdded1 = self.distanceMatrix[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + self.distanceMatrix[b2.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                            costRemoved2 = self.distanceMatrix[a2.ID][b2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + self.distanceMatrix[b2.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                            costAdded2 = self.distanceMatrix[a2.ID][b1.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + self.distanceMatrix[b1.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)

                            costChangeFirstRoute = costAdded1 - costRemoved1
                            costChangeSecondRoute = costAdded2 - costRemoved2

                            moveCost = costAdded1 + costAdded2 - (costRemoved1 + costRemoved2)

                        if moveCost < sm.moveCost:
                            self.StoreBestSwapMove(firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex,
                                                   moveCost, costChangeFirstRoute, costChangeSecondRoute, sm)


    