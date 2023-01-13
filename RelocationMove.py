from VRP_Model import *

class RelocationMove(object):
    def __init__(self):
        self.originRoutePosition = None
        self.targetRoutePosition = None
        self.originNodePosition = None
        self.targetNodePosition = None
        self.costChangeOriginRt = None
        self.costChangeTargetRt = None
        self.moveCost = None
        self.moveCost_penalized = None

    def Initialize(self):
        self.originRoutePosition = None
        self.targetRoutePosition = None
        self.originNodePosition = None
        self.targetNodePosition = None
        self.costChangeOriginRt = None
        self.costChangeTargetRt = None
        self.moveCost = 10 ** 9
        self.moveCost_penalized = 10 ** 9

def FindBestRelocationMove(self, rm):
        for originRouteIndex in range(0, len(self.sol.routes)):
            rt1: Route = self.sol.routes[originRouteIndex]
            rt1.sequenceOfNodes.append(self.dummy)
            for originNodeIndex in range(1, len(rt1.sequenceOfNodes) - 1):
                for targetRouteIndex in range(0, len(self.sol.routes)):
                    rt2: Route = self.sol.routes[targetRouteIndex]
                    rt2.sequenceOfNodes.append(self.dummy)
                    for targetNodeIndex in range(0, len(rt2.sequenceOfNodes) - 1):

                        if originRouteIndex == targetRouteIndex and (
                                targetNodeIndex == originNodeIndex or targetNodeIndex == originNodeIndex - 1):
                            continue

                        A = rt1.sequenceOfNodes[originNodeIndex - 1]
                        B = rt1.sequenceOfNodes[originNodeIndex]
                        C = rt1.sequenceOfNodes[originNodeIndex + 1]
                        len1 = len(rt1.sequenceOfNodes)
                        indexC = originNodeIndex + 1

                        F = rt2.sequenceOfNodes[targetNodeIndex]
                        G = rt2.sequenceOfNodes[targetNodeIndex + 1]
                        len2 = len(rt2.sequenceOfNodes)
                        indexG = targetNodeIndex + 1

                        if rt1 != rt2:
                            if rt2.load + B.demand > rt2.capacity:
                                continue
                        
                        costRemoved_penalized = (self.distance_matrix_penalized[A.ID][B.ID] + B.unload + \
                                                 self.distance_matrix_penalized[B.ID][C.ID])*(len1-indexC) + \
                                                 self.distance_matrix_penalized[F.ID][G.ID]*(len2-indexG) + rt1.cumMatrix[originNodeIndex]

                        costAdded_penalized = self.distance_matrix_penalized[A.ID][C.ID]*(len1 - indexC) +\
                                    (self.distance_matrix_penalized[F.ID][B.ID] + B.unload + self.distance_matrix_penalized[B.ID][G.ID])*(len2-indexG) +\
                                    + rt2.cumMatrix[targetNodeIndex] + F.unload + self.distance_matrix_penalized[F.ID][B.ID]

                        moveCost_penalized = costAdded_penalized - costRemoved_penalized

                        if (moveCost_penalized < rm.moveCost_penalized):
                            self.StoreBestRelocationMove(originRouteIndex, targetRouteIndex, originNodeIndex,
                                                         targetNodeIndex, moveCost_penalized, rm)
                    rt2.sequenceOfNodes.pop()
            rt1.sequenceOfNodes.pop()

def StoreBestRelocationMove(self, originRouteIndex, targetRouteIndex, originNodeIndex, targetNodeIndex,
                            moveCost_penalized, rm: RelocationMove):
    rm.originRoutePosition = originRouteIndex
    rm.originNodePosition = originNodeIndex
    rm.targetRoutePosition = targetRouteIndex
    rm.targetNodePosition = targetNodeIndex
    rm.moveCost_penalized = moveCost_penalized



def ApplyRelocationMove(self, rm: RelocationMove):

    originRt = self.sol.routes[rm.originRoutePosition]
    targetRt = self.sol.routes[rm.targetRoutePosition]

    B = originRt.sequenceOfNodes[rm.originNodePosition]

    if originRt == targetRt:
        del originRt.sequenceOfNodes[rm.originNodePosition]
        if (rm.originNodePosition < rm.targetNodePosition):
            targetRt.sequenceOfNodes.insert(rm.targetNodePosition, B)
        else:
            targetRt.sequenceOfNodes.insert(rm.targetNodePosition + 1, B)
    else:
        del originRt.sequenceOfNodes[rm.originNodePosition]
        targetRt.sequenceOfNodes.insert(rm.targetNodePosition + 1, B)
        originRt.load -= B.demand
        targetRt.load += B.demand

    self.sol.cost = self.CalculateCumCost(self.sol)

def InitializeOperators(self, rm, sm, top):
    rm.Initialize()
    sm.Initialize()
    top.Initialize()

def CalculateCumCost(self, sol):
    totalSolCost = 0
    for r in range(0, len(self.sol.routes)):
        rt: Route = self.sol.routes[r]
        rtCost = 0
        rtCumCost = 0
        for n in range(len(rt.sequenceOfNodes) - 2): #an exeis vgalei to dummy otan eisai edw allakse tous deiktes
            A = rt.sequenceOfNodes[n]
            B = rt.sequenceOfNodes[n + 1]
            rtCost += self.distance_matrix[A.ID][B.ID]
            rtCumCost += rtCost
            rtCost += B.unload
        totalSolCost += rtCumCost
    return totalSolCost

def penalize_arcs(self):
    # if self.penalized_n1_ID != -1 and self.penalized_n2_ID != -1:
    #     self.distance_matrix_penalized[self.penalized_n1_ID][self.penalized_n2_ID] = self.distance_matrix[self.penalized_n1_ID][self.penalized_n2_ID]
    #     self.distance_matrix_penalized[self.penalized_n2_ID][self.penalized_n1_ID] = self.distance_matrix[self.penalized_n2_ID][self.penalized_n1_ID]
    max_criterion = 0
    pen_1 = -1
    pen_2 = -1
    for i in range(len(self.sol.routes)):
        rt = self.sol.routes[i]
        for j in range(len(rt.sequenceOfNodes) - 1):
            id1 = rt.sequenceOfNodes[j].ID
            id2 = rt.sequenceOfNodes[j + 1].ID
            criterion = self.distance_matrix[id1][id2] / (1 + self.times_penalized[id1][id2])
            if criterion > max_criterion:
                max_criterion = criterion
                pen_1 = id1
                pen_2 = id2
    self.times_penalized[pen_1][pen_2] += 1
    self.times_penalized[pen_2][pen_1] += 1

    pen_weight = 0.15

    self.distance_matrix_penalized[pen_1][pen_2] = (1 + pen_weight * self.times_penalized[pen_1][pen_2]) * self.distance_matrix[pen_1][pen_2]
    self.distance_matrix_penalized[pen_2][pen_1] = (1 + pen_weight * self.times_penalized[pen_2][pen_1]) * self.distance_matrix[pen_2][pen_1]
    self.penalized_n1_ID = pen_1
    self.penalized_n2_ID = pen_2

