# https://trace.tennessee.edu/cgi/viewcontent.cgi?article=6358&context=utk_gradthes#:~:text=The%20genetic%20algorithm%20was%20developed,routing%20problems%2C%20and%20many%20others.

import time
from math import sqrt
from random import randint, seed, choice


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
        self.total_cost = sum(r.total for r in routes)
        self.total_cumulative_cost = sum(r.cumulative_cost for r in routes)

    def clone_solution(self, depot, capacity, distance_matrix):
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
            cloned_route.cumulative_cost = calculate_route_cumulative(cloned_route.sequenceOfNodes, distance_matrix)
            cloned_routes.append(cloned_route)
        cloned = Solution(cloned_routes)
        cloned.total_cost = self.total_cost
        return cloned

    def calculate_cumulative(self):
        self.total_cumulative_cost = sum(rt.cumulative_cost for rt in self.routes)


class TwoOptMove:
    """
    Represents a 2-Opt Move used in Local Search. Contains attributes for the necessary positions of both routes and
    nodes, as well as the change in the cost of a move.
    """

    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = None

    def initialize(self):
        """
        :return: None
        Initializes the 2-Opt object.
        """
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = 10 ** 9


class Tabu:
    """
    A Tabu objects contains the range of iterations in which nodes are tabooed.
    """

    def __init__(self, minimum, maximum):
        self.min_tenure = minimum
        self.max_tenure = maximum

    @staticmethod
    def move_is_tabu(current_solution: Solution, best_solution: Solution, node: Node, move_cost, iterator, eps=0.001):
        """
        :param current_solution: the solution to check if the move is tabooed
        :param best_solution: the best solution calculated thus far
        :param node: a Node object to check if it is tabooed
        :param move_cost: the move's change in cost
        :param iterator: an integer that represents the iteration Tabu search is at
        :param eps: a very small number epsilon
        :return: True if the move is tabooed and False otherwise or if the move leads to a better solution nonetheless
        """
        # if current_solution.total_cumulative_cost + move_cost < best_solution.total_cumulative_cost - eps:
        if current_solution.total_cost + move_cost < best_solution.total_cost - eps:
            return False
        return iterator < node.isTabuTillIterator

    def set_tabu(self, tabu_node: Node, iterator):
        """
        :param tabu_node: the Node object to be tabooed
        :param iterator: the iteration Tabu search is at
        :return: None
        Assigns a random integer in the range [iterator + min_tenure, iterator + max_tenure] to determine for how many
        iterations this node will be tabooed.
        """
        tabu_node.isTabuTillIterator = iterator + randint(self.min_tenure, self.max_tenure)


def read_vrp_problem_info():
    with open('../Instance.txt') as f:
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
    with open('../my_solution.txt', 'w') as f:
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
    cloned.total = rt.total
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
    # slice with the nodes from position top.positionOfFirstNode + 1 onwards
    relocated_segment_of_route_1 = node_sequence_route_1[index_of_node_1 + 1:]

    node_sequence_route_2 = route_2.sequenceOfNodes[:]
    # slice with the nodes from position top.positionOfFirstNode + 1 onwards
    relocated_segment_of_route_2 = node_sequence_route_2[index_of_node_2 + 1:]

    del node_sequence_route_1[index_of_node_1 + 1:]
    del node_sequence_route_2[index_of_node_2 + 1:]

    node_sequence_route_1.extend(relocated_segment_of_route_2)
    node_sequence_route_2.extend(relocated_segment_of_route_1)

    cost_added = calculate_route_cumulative(node_sequence_route_1, matrix) + \
                 calculate_route_cumulative(node_sequence_route_2, matrix)
    return cost_added - cost_removed


def calculate_intra_route_change(index_of_node_1, index_of_node_2, route: Route, matrix):
    node_a = route.sequenceOfNodes[index_of_node_1]
    node_b = route.sequenceOfNodes[index_of_node_1 + 1]
    node_k = route.sequenceOfNodes[index_of_node_2]
    node_l = route.sequenceOfNodes[index_of_node_2 + 1]
    cost_added = matrix[node_a.ID][node_k.ID] + matrix[node_b.ID][node_l.ID]
    cost_removed = matrix[node_a.ID][node_b.ID] + matrix[node_k.ID][node_l.ID]
    return cost_added - cost_removed


def calculate_inter_route_change(index_of_node_1, index_of_node_2, route_1, route_2, matrix):
    node_a = route_1.sequenceOfNodes[index_of_node_1]
    node_b = route_1.sequenceOfNodes[index_of_node_1 + 1]
    node_k = route_2.sequenceOfNodes[index_of_node_2]
    node_l = route_2.sequenceOfNodes[index_of_node_2 + 1]
    cost_added = matrix[node_a.ID][node_l.ID] + matrix[node_k.ID][node_b.ID]
    cost_removed = matrix[node_a.ID][node_b.ID] + matrix[node_k.ID][node_l.ID]
    return cost_added - cost_removed


def find_best_two_opt_move(top, sol: Solution, best_sol: Solution, iteration, matrix):
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
                        # move_cost = calculate_intra_route_cumulative_change(
                        #     index_of_node_1, index_of_node_2, route_1, matrix)
                        move_cost = calculate_intra_route_change(index_of_node_1, index_of_node_2, route_1, matrix)
                    else:
                        if index_of_node_1 == 0 and index_of_node_2 == 0:
                            continue
                        if index_of_node_1 == len(route_1.sequenceOfNodes) - 2 and index_of_node_2 == len(
                                route_2.sequenceOfNodes) - 2:
                            continue

                        if capacity_is_violated(route_1, index_of_node_1, route_2, index_of_node_2):
                            continue
                        # move_cost = calculate_inter_route_cumulative_change(index_of_node_1, index_of_node_2, route_1,
                        #                                                     route_2, matrix)
                        move_cost = calculate_inter_route_change(index_of_node_1, index_of_node_2, route_1,
                                                                 route_2, matrix)

                    if Tabu.move_is_tabu(sol, best_sol, route_1.sequenceOfNodes[index_of_node_1], move_cost,
                                         iteration) or \
                            Tabu.move_is_tabu(sol, best_sol, route_2.sequenceOfNodes[index_of_node_2], move_cost,
                                              iteration):
                        continue

                    if move_cost < top.moveCost:
                        store_best_two_opt_move(index_of_route_1, index_of_route_2, index_of_node_1, index_of_node_2,
                                                move_cost, top)


def capacity_is_violated(route_1, index_of_node_1, route_2, index_of_node_2):
    load_of_route_1_first_segment = sum(route_1.sequenceOfNodes[i].demand for i in range(0, index_of_node_1 + 1))
    load_of_route_1_second_segment = route_1.load - load_of_route_1_first_segment

    load_of_route_2_first_segment = sum(route_2.sequenceOfNodes[i].demand for i in range(0, index_of_node_2 + 1))
    load_of_route_2_second_segment = route_2.load - load_of_route_2_first_segment

    route_1_violated = load_of_route_1_first_segment + load_of_route_2_second_segment > route_1.capacity
    route_2_violated = load_of_route_2_first_segment + load_of_route_1_second_segment > route_2.capacity

    return route_1_violated or route_2_violated


def store_best_two_opt_move(index_of_route_1, index_of_route_2, index_of_node_1, index_of_node_2, move_cost, top):
    top.positionOfFirstRoute = index_of_route_1
    top.positionOfSecondRoute = index_of_route_2
    top.positionOfFirstNode = index_of_node_1
    top.positionOfSecondNode = index_of_node_2
    top.moveCost = move_cost


def apply_two_opt_move(top, sol: Solution, tabu: Tabu, iteration, matrix):
    route_1: Route = sol.routes[top.positionOfFirstRoute]
    route_2: Route = sol.routes[top.positionOfSecondRoute]
    node_1 = route_1.sequenceOfNodes[top.positionOfFirstNode]
    node_2 = route_2.sequenceOfNodes[top.positionOfSecondNode]

    if route_1 == route_2:
        # reverses the nodes in the segment [positionOfFirstNode + 1,  top.positionOfSecondNode]
        reversed_segment = reversed(route_1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1])
        # lst = list(reversed_segment)
        # lst2 = list(reversed_segment)
        route_1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversed_segment

        # reversedSegmentList =
        # list(reversed(route_1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1]))
        # route_1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegmentList

        # route_1.cumulative_cost += top.moveCost
        route_1.total += top.moveCost

    else:
        # slice with the nodes from position top.positionOfFirstNode + 1 onwards
        relocated_segment_of_route_1 = route_1.sequenceOfNodes[top.positionOfFirstNode + 1:]
        load_of_relocated_route_1 = sum(node.demand for node in relocated_segment_of_route_1)
        load_of_remaining_route_1 = route_1.load - load_of_relocated_route_1

        # slice with the nodes from position top.positionOfFirstNode + 1 onwards
        relocated_segment_of_route_2 = route_2.sequenceOfNodes[top.positionOfSecondNode + 1:]
        load_of_relocated_route_2 = sum(node.demand for node in relocated_segment_of_route_2)
        load_of_remaining_route_2 = route_2.load - load_of_relocated_route_2

        del route_1.sequenceOfNodes[top.positionOfFirstNode + 1:]
        del route_2.sequenceOfNodes[top.positionOfSecondNode + 1:]

        route_1.sequenceOfNodes.extend(relocated_segment_of_route_2)
        route_2.sequenceOfNodes.extend(relocated_segment_of_route_1)

        route_1.load = load_of_remaining_route_1 + load_of_relocated_route_2
        route_2.load = load_of_remaining_route_2 + load_of_relocated_route_1

        # update_route_cost_and_load(route_1, matrix)
        # update_route_cost_and_load(route_2, matrix)

    # sol.total_cumulative_cost += top.moveCost
    sol.total_cost += top.moveCost
    tabu.set_tabu(node_1, iteration)
    tabu.set_tabu(node_2, iteration)


def update_route_cost_and_load(route: Route, matrix):
    route.cumulative_cost = calculate_route_cumulative(route.sequenceOfNodes, matrix)
    route.load = sum(node.demand for node in route.sequenceOfNodes)


def tabu_search(initial_solution, tenure_range, iterations, matrix, depot, capacity):
    """
    :param initial_solution: the initial solution constructed
    :param tenure_range: how much will each taboo lasts
    :param iterations: number of iterations to tabu search
    :param matrix: cost-distance matrix
    :param depot: initial node where each route starts from
    :param capacity: max load of each vehicle
    :return: the best solution found in the search procedure
    Implementation of Tabu Search using only 2-Opt Operator
    """
    solution = initial_solution.clone_solution(depot, capacity, matrix)
    best_solution = initial_solution.clone_solution(depot, capacity, matrix)
    top = TwoOptMove()
    tabu = Tabu(*tenure_range)  # tuple unpacking

    for iterator in range(iterations):
        top.initialize()
        find_best_two_opt_move(top, solution, best_solution, iterator, matrix)
        apply_two_opt_move(top, solution, tabu, iterator, matrix)
        # if solution.total_cumulative_cost < best_solution.total_cumulative_cost:
        if solution.total_cost < best_solution.total_cost:
            best_solution = solution.clone_solution(depot, capacity, matrix)
        print(f'Iteration: {iterator}, Solution cost: {solution.total_cost},'
              f' best Solution cost: {best_solution.total_cost}')
    return Solution(best_solution.routes)


def construct_solution(depot, capacity, vehicles, nodes, matrix):
    routes = [Route(depot, capacity) for _ in range(vehicles)]
    unserved = nodes[:]
    while unserved:
        route_pos = -1
        node_pos = -1
        # min_cum = 10 ** 10
        min_cost = 10 ** 10
        for pos_n, node in enumerate(unserved):
            for pos_r, route in enumerate(routes):
                from_node = route.last()
                cost = matrix[from_node.ID][node.ID]
                total = route.total + cost
                # if total < min_cum and route.load + node.demand <= route.capacity:
                if total < min_cost and route.load + node.demand <= route.capacity:
                    # min_cum = total
                    min_cost = cost
                    route_pos = pos_r
                    node_pos = pos_n
        node_to_be_added = unserved.pop(node_pos)
        routes[route_pos].add_node(node_to_be_added, min_cost)
    return Solution(routes)


def main():
    start_time = time.time()

    number_of_vehicles, capacity, customers, nodes, depot = read_vrp_problem_info()
    seed(1)

    distance_matrix = calculate_cost_matrix([depot] + nodes)

    initial_solution = construct_solution(depot, capacity, number_of_vehicles, nodes, distance_matrix)

    number_of_iterations = 2000
    tenure_range = (10, 20)
    best_solution = tabu_search(initial_solution, tenure_range, number_of_iterations, distance_matrix,
                                depot, capacity)
    write_solution_to_file(best_solution, number_of_vehicles)

    end_time = time.time()
    print(f'time elapsed: {end_time - start_time}')


if __name__ == '__main__':
    main()
