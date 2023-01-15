import time
from math import sqrt
from random import randint, seed, shuffle, randrange


class Node:
    """
    Node is actually representing a customer in a VRP. Apart from its id and coordinates, it is also characterised by
    the demand, time to unload, and for the purposes of tabu search a variable until the iteration this node cannot be
    used unless it brings a better solution.
    """

    def __init__(self, id_n=0, x_cor=0, y_cor=0, dem=0, un_time=0):
        self.ID = id_n
        self.x = x_cor
        self.y = y_cor
        self.demand = dem
        self.unloading_time = un_time
        self.isTabuTillIterator = -1

    def from_line(self, line, sep=','):
        """
        :param line: a single file_line from a text file
        :param sep: separator(aka delimiter) used in that file to divide values(',' in csv files)
        :return: None
        The nodes' csv file has the following form: id, x, y, demand, unloading_time
        This method splits the file_line on the delimiter sep and then initializes the node's values
        """
        split_line = line.split(sep)
        self.ID = int(split_line[0])
        self.x = int(split_line[1])
        self.y = int(split_line[2])
        self.demand = int(split_line[3])
        self.unloading_time = int(split_line[4])

    def __eq__(self, other):
        return self.ID == other.ID

    def __str__(self):
        return f'Node({self.ID}, {self.x}, {self.y}, {self.demand}, {self.unloading_time})'

    def __repr__(self):
        return str(self)


class Route:
    """
    Route class represents a route in a VRP. Apart from the sequence of nodes, it holds information about the total
    cost, as well as the cumulative cost(objective of this problem). It also contains the current load of that route
    and the maximum capacity it can hold(load <= capacity).
    """

    def __init__(self, dp, cap):
        self.sequenceOfNodes = [dp]
        self.total = 0
        self.cumulative_cost = 0
        self.load = 0
        self.capacity = cap

    def add_node(self, new_node: Node, minimum_cost):
        """
        :param new_node: node to be added in the sequence of nodes
        :param minimum_cost: cost added to the total
        :return: None
        When a new node is added in the route, it enters the sequence of nodes list, and both the cumulative
        cost and total load change. The feasibility of a new node addition is not checked in this method.
        """
        self.sequenceOfNodes.append(new_node)
        self.total += minimum_cost
        self.cumulative_cost += self.total
        self.total += new_node.unloading_time
        self.load += new_node.demand

    def last(self):
        """
        :return: the last node in the sequence of nodes list
        """
        return self.sequenceOfNodes[-1]

    def __str__(self):
        return ' - '.join(list(str(node.ID) for node in self.sequenceOfNodes))

    def __repr__(self):
        return str(self)


class Solution:
    """
    Solution class for the VRP. contains the routes and the total cumulative cost.
    """

    def __init__(self, routes: list):
        self.routes = routes
        self.total_cumulative_cost = sum(r.cumulative_cost for r in routes)

    def clone(self, depot, capacity):
        """
        :param depot: initial node of route
        :param capacity: maximum load of vehicles
        :return: a copied version of this solution
        Deep copies this current solution to another Solution object.
        """
        cloned_routes = []
        for route_ind in range(0, len(self.routes)):
            rt = self.routes[route_ind]
            cloned_route = clone_route(rt, depot, capacity)
            cloned_routes.append(cloned_route)
        cloned = Solution(cloned_routes)
        return cloned

    def perturbate(self, matrix):
        flag = True
        while flag:
            sm = SwapMove()
            sm.position_of_original_route = randrange(len(self.routes))
            sm.position_of_target_route = randrange(len(self.routes))
        
            route_1 = self.routes[sm.position_of_original_route]
            route_2 = self.routes[sm.position_of_target_route]
        
            sm.position_of_original_node = randrange(1, len(route_1.sequenceOfNodes))
            sm.position_of_target_node = randrange(1, len(route_2.sequenceOfNodes))

            if sm.position_of_original_route == sm.position_of_target_route:
                if sm.position_of_original_node == sm.position_of_target_node:
                    continue
                sm.move_cost = calculate_intra_swap_cumulative_change(
                    sm.position_of_original_node, sm.position_of_target_node, route_1, matrix)
            else:
                node_sequence_route_1 = route_1.sequenceOfNodes[:]
                node_sequence_route_2 = route_2.sequenceOfNodes[:]
                (node_sequence_route_1[sm.position_of_original_node], node_sequence_route_2[sm.position_of_target_node]) = (
                    node_sequence_route_2[sm.position_of_target_node], node_sequence_route_1[sm.position_of_original_node])
                rt1_demand = sum(i.demand for i in node_sequence_route_1)
                rt2_demand = sum(i.demand for i in node_sequence_route_2)

                if rt1_demand > route_1.capacity or rt2_demand > route_2.capacity:
                    continue

                sm.move_cost = calculate_inter_swap_cumulative_change(sm.position_of_original_node,
                                                                      sm.position_of_target_node, route_1, route_2,
                                                                      matrix)

            sm.apply_move(self, matrix)
            flag = False


class Move:
    def __init__(self):
        self.position_of_original_route = None
        self.position_of_target_route = None
        self.position_of_original_node = None
        self.position_of_target_node = None
        self.move_cost = None

    def initialize(self):
        self.position_of_original_route = None
        self.position_of_target_route = None
        self.position_of_original_node = None
        self.position_of_target_node = None
        self.move_cost = 10 ** 9

    def store_best_move(self, route_1, route_2, node_1, node_2, cost):
        self.position_of_original_route = route_1
        self.position_of_target_route = route_2
        self.position_of_original_node = node_1
        self.position_of_target_node = node_2
        self.move_cost = cost

    def find_best_move(self, sol, matrix):
        pass

    def apply_move(self, sol, matrix):
        pass

    def __str__(self):
        return 'Move'

    def __repr__(self):
        return str(self)


class TwoOptMove(Move):
    """
    Represents a 2-Opt Move used in Local Search. Contains attributes for the necessary positions of both routes and
    nodes, as well as the change in the cost of a move.
    """

    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()

    def find_best_move(self, sol, matrix):
        for index_of_route_1 in range(0, len(sol.routes)):
            route_1: Route = sol.routes[index_of_route_1]
            for index_of_route_2 in range(index_of_route_1, len(sol.routes)):
                route_2: Route = sol.routes[index_of_route_2]
                for index_of_node_1 in range(0, len(route_1.sequenceOfNodes) - 1):
                    start2 = 0
                    if route_1 == route_2:
                        start2 = index_of_node_1 + 2
                    for index_of_node_2 in range(start2, len(route_2.sequenceOfNodes) - 1):

                        if route_1 == route_2:
                            if index_of_node_1 == 0 and index_of_node_2 == len(route_1.sequenceOfNodes) - 2:
                                continue
                            move_cost = calculate_intra_route_cumulative_change(
                                index_of_node_1, index_of_node_2, route_1, matrix)
                        else:
                            if index_of_node_1 == 0 and index_of_node_2 == 0:
                                continue
                            if index_of_node_1 == len(route_1.sequenceOfNodes) - 2 and index_of_node_2 == len(
                                    route_2.sequenceOfNodes) - 2:
                                continue

                            if capacity_is_violated(route_1, index_of_node_1, route_2, index_of_node_2):
                                continue
                            move_cost = calculate_inter_route_cumulative_change(index_of_node_1, index_of_node_2,
                                                                                route_1,
                                                                                route_2, matrix)

                        if move_cost < self.move_cost:
                            self.store_best_move(index_of_route_1, index_of_route_2, index_of_node_1, index_of_node_2,
                                                 move_cost)

    def apply_move(self, sol, matrix):
        route_1: Route = sol.routes[self.position_of_original_route]
        route_2: Route = sol.routes[self.position_of_target_route]

        if route_1 == route_2:
            reversed_segment = reversed(
                route_1.sequenceOfNodes[self.position_of_original_node + 1: self.position_of_target_node + 1])
            route_1.sequenceOfNodes[
            self.position_of_original_node + 1: self.position_of_target_node + 1] = reversed_segment
            route_1.cumulative_cost += self.move_cost

        else:
            relocated_segment_of_route_1 = route_1.sequenceOfNodes[self.position_of_original_node + 1:]

            relocated_segment_of_route_2 = route_2.sequenceOfNodes[self.position_of_target_node + 1:]

            del route_1.sequenceOfNodes[self.position_of_original_node + 1:]
            del route_2.sequenceOfNodes[self.position_of_target_node + 1:]

            route_1.sequenceOfNodes.extend(relocated_segment_of_route_2)
            route_2.sequenceOfNodes.extend(relocated_segment_of_route_1)

            update_route_cost_and_load(route_1, matrix)
            update_route_cost_and_load(route_2, matrix)

        sol.total_cumulative_cost += self.move_cost

    def __str__(self):
        return '2-Opt'

    def __repr__(self):
        return str(self)


class SwapMove(Move):
    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()

    def find_best_move(self, sol, matrix):
        for firstRouteIndex in range(0, len(sol.routes)):
            rt1: Route = sol.routes[firstRouteIndex]
            for secondRouteIndex in range(firstRouteIndex, len(sol.routes)):
                rt2: Route = sol.routes[secondRouteIndex]
                for firstNodeIndex in range(1, len(rt1.sequenceOfNodes)):
                    startOfSecondNodeIndex = 1
                    if rt1 == rt2:
                        startOfSecondNodeIndex = firstNodeIndex + 1
                    for secondNodeIndex in range(startOfSecondNodeIndex, len(rt2.sequenceOfNodes)):

                        if rt1 == rt2:

                            moveCost = calculate_intra_swap_cumulative_change(firstNodeIndex, secondNodeIndex, rt1,
                                                                              matrix)

                        else:
                            node_sequence_route_1 = rt1.sequenceOfNodes[:]
                            node_sequence_route_2 = rt2.sequenceOfNodes[:]
                            (node_sequence_route_1[firstNodeIndex], node_sequence_route_2[secondNodeIndex]) = (
                                node_sequence_route_2[secondNodeIndex], node_sequence_route_1[firstNodeIndex])
                            rt1_demand = sum(i.demand for i in node_sequence_route_1)
                            rt2_demand = sum(i.demand for i in node_sequence_route_2)

                            if rt1_demand > rt1.capacity or rt2_demand > rt2.capacity:
                                continue

                            moveCost = calculate_inter_swap_cumulative_change(firstNodeIndex, secondNodeIndex, rt1, rt2,
                                                                              matrix)

                        if moveCost < self.move_cost:
                            self.store_best_move(firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex,
                                                 moveCost)

    def apply_move(self, sol, matrix):
        rt1 = sol.routes[self.position_of_original_route]
        rt2 = sol.routes[self.position_of_target_route]
        b1 = rt1.sequenceOfNodes[self.position_of_original_node]
        b2 = rt2.sequenceOfNodes[self.position_of_target_node]
        rt1.sequenceOfNodes[self.position_of_original_node] = b2
        rt2.sequenceOfNodes[self.position_of_target_node] = b1

        if rt1 == rt2:
            rt1.cumulative_cost += self.move_cost
        else:

            update_route_cost_and_load(rt1, matrix)
            update_route_cost_and_load(rt2, matrix)
        sol.total_cumulative_cost += self.move_cost

    def __str__(self):
        return 'Swap'

    def __repr__(self):
        return str(self)


class RelocationMove(Move):
    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()

    def find_best_move(self, sol, matrix):
        for originRouteIndex in range(0, len(sol.routes)):
            rt1: Route = sol.routes[originRouteIndex]
            for originNodeIndex in range(1, len(rt1.sequenceOfNodes)):
                for targetRouteIndex in range(0, len(sol.routes)):
                    rt2: Route = sol.routes[targetRouteIndex]
                    for targetNodeIndex in range(0, len(rt2.sequenceOfNodes)):
                        if rt1 != rt2:
                            current_cap = sum([i.demand for i in rt2.sequenceOfNodes])
                            if current_cap + rt1.sequenceOfNodes[originNodeIndex].demand > rt2.capacity:
                                continue
                            move_cost = calculate_inter_relocation_cumulative_change(originNodeIndex, targetNodeIndex,
                                                                                     rt1,
                                                                                     rt2, matrix)
                        else:
                            move_cost = calculate_intra_relocation_cumulative_change(originNodeIndex, targetNodeIndex,
                                                                                     rt1,
                                                                                     matrix)

                        if move_cost < self.move_cost:
                            self.store_best_move(originRouteIndex, targetRouteIndex, originNodeIndex, targetNodeIndex,
                                                 move_cost)

    def apply_move(self, sol, matrix):
        originRt = sol.routes[self.position_of_original_route]
        targetRt = sol.routes[self.position_of_target_route]

        B = originRt.sequenceOfNodes[self.position_of_original_node]

        if originRt == targetRt:
            del originRt.sequenceOfNodes[self.position_of_original_node]
            if self.position_of_original_node < self.position_of_target_node:
                targetRt.sequenceOfNodes.insert(self.position_of_target_node, B)
            else:
                targetRt.sequenceOfNodes.insert(self.position_of_target_node + 1, B)
            originRt.cumulative_cost += self.move_cost
        else:
            del originRt.sequenceOfNodes[self.position_of_original_node]
            targetRt.sequenceOfNodes.insert(self.position_of_target_node + 1, B)
            originRt.load -= B.demand
            targetRt.load += B.demand
            update_route_cost_and_load(originRt, matrix)
            update_route_cost_and_load(targetRt, matrix)
        sol.total_cumulative_cost += self.move_cost

    def __str__(self):
        return 'Relocation'

    def __repr__(self):
        return str(self)


def read_vrp_problem_info():
    with open('Instance.txt') as f:
        number_of_vehicles = get_int_from_line(f.readline())
        capacity = get_int_from_line(f.readline())
        customers = get_int_from_line(f.readline())
        f.readline()  # get rid of NODE INFO
        f.readline()  # get rid of column names
        depot = Node()
        depot.from_line(f.readline())
        nodes = []
        for line in f.readlines():
            node = Node()
            node.from_line(line)
            nodes.append(node)
    return number_of_vehicles, capacity, customers, nodes, depot


def write_solution_to_file(solution, number_of_vehicles):
    with open('my_solution.txt', 'w') as f:
        f.write(f'Cost:\n{solution.total_cumulative_cost}\n')
        f.write(f'Routes:\n{number_of_vehicles}')
        for route in solution.routes:
            f.write('\n')
            for node in route.sequenceOfNodes:
                f.write(f'{node.ID}')
                if node != route.last():
                    f.write(',')


def get_int_from_line(file_line, pos=1, sep=','):
    """
    :param file_line: a line from a file
    :param pos: the position in which we want to extract the integer
    :param sep: the delimiter used in the file(e.g. ',' in csv files)
    :return: an integer
    """
    return int(file_line.split(sep)[pos])


def calculate_euclid(node_from, node_to):
    """
    :param node_from: Node object we start from
    :param node_to: Node object we end up
    :return: euclidean distance of nodes
    """
    return sqrt((node_from.x - node_to.x) ** 2 + (node_from.y - node_to.y) ** 2)


def calculate_cost_matrix(sequence_of_nodes):
    """
    :param sequence_of_nodes: sequence of customer Nodes
    :return: a List of List objects containing the cost from one node to another for each node
    """
    matrix = [[0 for _ in range(len(sequence_of_nodes))] for _ in range(len(sequence_of_nodes))]
    for from_index in range(len(sequence_of_nodes)):
        for to_index in range(from_index + 1, len(sequence_of_nodes)):
            from_n = sequence_of_nodes[from_index]
            to_n = sequence_of_nodes[to_index]
            cost = calculate_euclid(from_n, to_n)
            matrix[from_index][to_index] = cost
            matrix[to_index][from_index] = cost
    return matrix


def clone_route(rt: Route, depot, capacity):
    cloned = Route(depot, capacity)
    cloned.cumulative_cost = rt.cumulative_cost
    cloned.load = rt.load
    cloned.sequenceOfNodes = rt.sequenceOfNodes.copy()
    return cloned


def calculate_route_cumulative(nodes_sequence, distance_matrix):
    rt_cumulative_cost = 0
    tot_time = 0
    for i in range(len(nodes_sequence) - 1):
        from_node: Node = nodes_sequence[i]
        to_node: Node = nodes_sequence[i + 1]
        tot_time += distance_matrix[from_node.ID][to_node.ID]
        rt_cumulative_cost += tot_time
        tot_time += to_node.unloading_time
    return rt_cumulative_cost


def calculate_intra_route_cumulative_change(index_of_node_1, index_of_node_2, route, matrix):
    cost_removed = route.cumulative_cost

    node_sequence_copy = route.sequenceOfNodes[:]
    reversed_segment = reversed(node_sequence_copy[index_of_node_1 + 1: index_of_node_2 + 1])
    node_sequence_copy[index_of_node_1 + 1: index_of_node_2 + 1] = reversed_segment

    cost_added = calculate_route_cumulative(node_sequence_copy, matrix)
    return cost_added - cost_removed


def calculate_inter_route_cumulative_change(index_of_node_1, index_of_node_2, route_1, route_2, matrix):
    cost_removed = route_1.cumulative_cost + route_2.cumulative_cost

    node_sequence_route_1 = route_1.sequenceOfNodes[:]
    # slice with the nodes from position top.position_of_original_node + 1 onwards
    relocated_segment_of_route_1 = node_sequence_route_1[index_of_node_1 + 1:]

    node_sequence_route_2 = route_2.sequenceOfNodes[:]
    # slice with the nodes from position top.position_of_original_node + 1 onwards
    relocated_segment_of_route_2 = node_sequence_route_2[index_of_node_2 + 1:]

    del node_sequence_route_1[index_of_node_1 + 1:]
    del node_sequence_route_2[index_of_node_2 + 1:]

    node_sequence_route_1.extend(relocated_segment_of_route_2)
    node_sequence_route_2.extend(relocated_segment_of_route_1)

    cost_added = calculate_route_cumulative(node_sequence_route_1, matrix) \
                 + calculate_route_cumulative(node_sequence_route_2, matrix)
    return cost_added - cost_removed


def calculate_inter_swap_cumulative_change(index_of_node_1, index_of_node_2, route_1, route_2, matrix):
    cost_removed = route_1.cumulative_cost + route_2.cumulative_cost

    node_sequence_route_1 = route_1.sequenceOfNodes[:]
    node_sequence_route_2 = route_2.sequenceOfNodes[:]
    (node_sequence_route_1[index_of_node_1], node_sequence_route_2[index_of_node_2]) = (
        node_sequence_route_2[index_of_node_2], node_sequence_route_1[index_of_node_1])

    cost_added = calculate_route_cumulative(node_sequence_route_1, matrix) \
                 + calculate_route_cumulative(node_sequence_route_2, matrix)
    return cost_added - cost_removed


def calculate_intra_swap_cumulative_change(index_of_node_1, index_of_node_2, route, matrix):
    cost_removed = route.cumulative_cost

    node_sequence_copy = route.sequenceOfNodes[:]
    (node_sequence_copy[index_of_node_1], node_sequence_copy[index_of_node_2]) = (
        node_sequence_copy[index_of_node_2], node_sequence_copy[index_of_node_1])

    cost_added = calculate_route_cumulative(node_sequence_copy, matrix)
    return cost_added - cost_removed


def calculate_intra_relocation_cumulative_change(index_of_node_1, index_of_node_2, route, matrix):
    cost_removed = route.cumulative_cost

    node_sequence_copy = route.sequenceOfNodes[:]
    temp = node_sequence_copy[index_of_node_1]
    del node_sequence_copy[index_of_node_1]
    if (index_of_node_1 < index_of_node_2):
        node_sequence_copy.insert(index_of_node_2, temp)
    else:
        node_sequence_copy.insert(index_of_node_2 + 1, temp)
    cost_added = calculate_route_cumulative(node_sequence_copy, matrix)
    return cost_added - cost_removed


def calculate_inter_relocation_cumulative_change(index_of_node_1, index_of_node_2, route_1, route_2, matrix):
    cost_removed = route_1.cumulative_cost + route_2.cumulative_cost

    node_sequence_route_1 = route_1.sequenceOfNodes[:]
    node_sequence_route_2 = route_2.sequenceOfNodes[:]
    temp = node_sequence_route_1[index_of_node_1]
    del node_sequence_route_1[index_of_node_1]
    node_sequence_route_2.insert(index_of_node_2 + 1, temp)

    cost_added = calculate_route_cumulative(node_sequence_route_1, matrix) \
                 + calculate_route_cumulative(node_sequence_route_2, matrix)
    return cost_added - cost_removed


def capacity_is_violated(route_1, index_of_node_1, route_2, index_of_node_2):
    load_of_route_1_first_segment = sum(route_1.sequenceOfNodes[i].demand for i in range(0, index_of_node_1 + 1))
    load_of_route_1_second_segment = route_1.load - load_of_route_1_first_segment

    load_of_route_2_first_segment = sum(route_2.sequenceOfNodes[i].demand for i in range(0, index_of_node_2 + 1))
    load_of_route_2_second_segment = route_2.load - load_of_route_2_first_segment

    route_1_violated = load_of_route_1_first_segment + load_of_route_2_second_segment > route_1.capacity
    route_2_violated = load_of_route_2_first_segment + load_of_route_1_second_segment > route_2.capacity

    return route_1_violated or route_2_violated


def construct_solution(depot, capacity, vehicles, customers, nodes, matrix):
    routes = [Route(depot, capacity) for _ in range(vehicles)]
    unserved = nodes[:]
    while unserved:
        route_pos = -1
        node_pos = -1
        min_cum = 10 ** 10
        min_cost = 10 ** 10
        for pos_n, node in enumerate(unserved):
            for pos_r, route in enumerate(routes):
                from_node = route.last()
                cost = matrix[from_node.ID][node.ID]
                total = route.total + cost
                if total < min_cum and route.load + node.demand <= route.capacity:
                # if total < min_cost and route.load + node.demand <= route.capacity:
                    min_cum = total
                    min_cost = cost
                    route_pos = pos_r
                    node_pos = pos_n
        node_to_be_added = unserved.pop(node_pos)
        routes[route_pos].add_node(node_to_be_added, min_cost)
    return Solution(routes)


def update_route_cost_and_load(route: Route, matrix):
    route.cumulative_cost = calculate_route_cumulative(route.sequenceOfNodes, matrix)
    route.load = sum(node.demand for node in route.sequenceOfNodes)


def initialize_moves():
    top = TwoOptMove()
    sm = SwapMove()
    rm = RelocationMove()
    ops = [top, sm, rm]
    return ops


def vns(initial_solution, matrix, depot, capacity, iterations):
    solution = initial_solution.clone(depot, capacity)
    best_solution = initial_solution.clone(depot, capacity)
    ops = initialize_moves()
    k_max = len(ops)

    for iteration in range(iterations):
        k = 0
        solution.perturbate(matrix)

        while k < k_max:
            operator = ops[k]

            operator.initialize()
            operator.find_best_move(solution, matrix)
            if operator.position_of_original_route is not None and operator.move_cost < 0:
                operator.apply_move(solution, matrix)
                k = 0
            else:
                k += 1

            if solution.total_cumulative_cost < best_solution.total_cumulative_cost:
                print('mphke', solution.total_cumulative_cost)
                best_solution = solution.clone(depot, capacity)
            print(f'Iteration: {iteration}, Solution cost: {solution.total_cumulative_cost},'
                  f' best Solution cost: {best_solution.total_cumulative_cost}',
                  f' Move Operator:{operator}')

    return best_solution


def main():
    start_time = time.time()

    number_of_vehicles, capacity, customers, nodes, depot = read_vrp_problem_info()
    seed(3)

    distance_matrix = calculate_cost_matrix([depot] + nodes)

    initial_best_solution = construct_solution(depot, capacity, number_of_vehicles, customers, nodes, distance_matrix)

    iterations = 40

    best_solution = vns(initial_best_solution, distance_matrix, depot, capacity, iterations)
    write_solution_to_file(best_solution, number_of_vehicles)

    end_time = time.time()
    print(f'time elapsed: {end_time - start_time}')


if __name__ == '__main__':
    main()
