# https://trace.tennessee.edu/cgi/viewcontent.cgi?article=6358&context=utk_gradthes#:~:text=The%20genetic%20algorithm%20was%20developed,routing%20problems%2C%20and%20many%20others.

import time
from math import sqrt
from random import randint, randrange, seed, random, shuffle, choice


class Node:
    """
    Node is actually representing a customer in a VRP. Apart from its id and coordinates, it is also characterised by
    the demand, time to unloading_time, and for the purposes of tabu search a variable until the iteration this node cannot be
    used unless it brings a better solution.
    """

    def __init__(self, id_n=0, x_cor=0, y_cor=0, dem=0, un_time=0):
        self.ID = id_n
        self.x = x_cor
        self.y = y_cor
        self.demand = dem
        self.unloading_time = un_time

    def from_line(self, line, sep=','):
        """
        :param line: a single file_line from a text file
        :param sep: separator(aka delimiter) used in that file to divide values(',' in csv files)
        :return: None
        The nodes' csv file has the following form: id, x, y, demand, unloading_timeing_time
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
        self.pen_cum_matrix = []
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

class Penalize:
    def __init__(self, rows, matrix):
        self.distance_matrix_penalized = [[matrix[i][j] for j in range(rows)] for i in range(rows)]
        self.times_penalized = [[0 for j in range(rows)] for i in range(rows)]
        self.penalized_n1_ID = -1
        self.penalized_n2_ID = -1
        
class Genome(list):
    """
    Genome class represents a solution of the problem used in a genetic algorithm. Instead of having a list of "nodes",
    Genome is a subclass of List class, thus inheriting all of its methods. Actually it doesn't hold Node objects, but a
    number that represents in which route the node belongs to and in a way its positioning(key). For example, suppose we
    have 2 routes and 6 nodes(excluding depot), and the genome is somewhat like this: [2.1, 1.7, 1.5, 2.5, 2.3, 1.1].
    This means, Route 1: Depot -> F -> C -> B, and Route 2: Depot -> A -> E -> D.
    This class also holds information about a particular gene's cumulative cost, total, a score(analyzed below) and
    whether it is a feasible solution to the problem or not. Score is a criterion used to sort the genome population.
    In practice, if the genome is a feasible solution to the problem, then score is equal to the cumulative cost.
    Otherwise the genome is penalized for not being feasible, making it go lower in the population.
    """

    def __init__(self):
        super().__init__()
        self.cumulative_cost = 0
        self.total = 0
        self.score = 0
        self.feasible = True

    def fitness(self, nodes, matrix, depot, capacity, number_of_vehicles):
        """
        :param nodes: a list that contains Node objects
        :param matrix: the cost matrix of the problem
        :param depot: a Node object representing the root of the route
        :param capacity: the maximum load a route can hold
        :param number_of_vehicles: how many vehicles to use in the problem
        :return: None
        This method calculates the fitness score of the particular genome. For the score calculation, the cumulative
        cost must be firstly calculate, using the nodes' list and the cost matrix, and then it must be checked if
        feasibility is violated in order to add a penalty equal to v * (t - v * c) ^ 2, where v is the number of
        vehicles, t the total cost of the route and c the capacity.
        """
        self.feasible = True
        node_sorted = [(customer, random_key) for customer, random_key in zip(nodes, self)]
        # combine the nodes with their keys, the random number assigned for each node, stored inside the object itself
        node_sorted += [(depot, vehicle + 1) for vehicle in range(number_of_vehicles)]  # add the depots needed
        # instead of a random key, depots have a fixed value since they are the start of every route
        node_sorted.sort(key=lambda x: x[1])  # sort the list based on the random key
        # the nodes are now in "order" and in terms of routes and positioning within the route
        cumulative = 0
        non_feasible_vehicles = 0  # number of routes that violate the capacity constraint
        route_total = 0
        route_cap = 0
        unchanged_feasibility = True  # used for every route in order not to add more vehicles in the counter
        prev = None
        for customer in node_sorted:
            # customer[0] is the Node object
            if customer[0] is depot:
                route_total = 0
                route_cap = 0
                unchanged_feasibility = True
            else:
                weight = matrix[customer[0].ID][prev[0].ID]
                route_total += weight
                self.total += weight
                cumulative += route_total
                route_total += customer[0].unloading_time
                route_cap += customer[0].demand
                if route_cap > capacity and unchanged_feasibility:
                    unchanged_feasibility = False  # set this to False to not increment counter for this route again
                    non_feasible_vehicles += 1
            prev = customer
        self.cumulative_cost = cumulative
        if non_feasible_vehicles > 0:  # that means the capacity constraint is violated, therefore penalize it
            self.feasible = False
            self.score = self.cumulative_cost + number_of_vehicles * (self.total - number_of_vehicles * capacity) ** 2
        else:
            self.score = self.cumulative_cost

    def crossover(self, other_genome, gene_selection_rate):
        """
        :param other_genome: another Genome object use for the crossover
        :param gene_selection_rate: the rate in which the first gene is selected over the other
        :return: None
        In this method, a genome is changed using the crossover process which involves the usage of a second genome
        selected from the population. With a given gene selection rate, for each gene in both genomes, one of them is
        chosen to pass into the next generation of that genome. A random number is chosen, and if it's < than the
        selection rate, then the gene doesn't change(chose first genome), otherwise the second genome's gene is
        selected.
        """
        for gene in range(len(self)):
            selection = random()
            if selection >= gene_selection_rate:
                self[gene] = other_genome[gene]

    def mutate(self, mutation_rate, number_of_vehicles):
        """
        :param mutation_rate: the chance in which a particular gene is mutated
        :param number_of_vehicles: the number of vehicles in the problem
        :return: None
        Every gene of a genome that mutates has a chance to change. The purpose of mutation is to diversify the
        population and prevent divergence to a population composed of 'elites' and therefore to a stagnated cost.
        A random number determines if a gene is mutated, and when it does, a vehicle is chosen(can be the same one)
        first, and then the in-route position of the gene.
        """
        for gene in range(len(self)):
            mutation = random()
            if mutation < mutation_rate:
                decimal = random()
                new_vehicle = randint(1, number_of_vehicles)
                self[gene] = new_vehicle + decimal

    def create_route_list(self, nodes, depot: Node, capacity, number_of_vehicles, distance_matrix):
        """
        :param nodes: sequence of Node objects
        :param depot: storage from where each route starts
        :param capacity: max load vehicles can load
        :param number_of_vehicles: how many vehicles used
        :param distance_matrix: distance-cost matrix
        :return: a List containing Route objects
        This method creates a list with Routes for a given genome. It creates Route objects, calculating any necessary
        information, and then appends each route to the list, before returning it.
        """
        routes = []
        # connect each node with its key value, add the depots and then sort by the key value
        node_sorted = [(node, value) for node, value in zip(nodes, self)]
        node_sorted += [(depot, i + 1) for i in range(number_of_vehicles)]
        node_sorted.sort(key=lambda x: x[1])
        for node, _ in node_sorted:
            if node == depot:
                routes.append(Route(depot, capacity))
            else:
                # because nodes were sorted by the key, there is no need to check it enters the correct Route object
                cost = distance_matrix[routes[-1].last().ID][node.ID]
                routes[-1].add_node(node, cost)
        return routes


class Solution:
    """
    Solution class for the VRP. contains the routes and the total cumulative cost.
    """

    def __init__(self, routes: list):
        self.routes = routes
        self.total_cumulative_cost = float(sum(r.cumulative_cost for r in routes))

    def clone_solution(self, depot, capacity):
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

class RelocationMove():
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
        self.moveCost_penalized = None

    def Initialize(self):
        """
        :return: None
        Initializes the 2-Opt object.
        """
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = 10 ** 9
        self.moveCost_penalized = 10 ** 9

class SwapMove:
    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = None
        self.moveCost_penalized = None

    def Initialize(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = 10 ** 9
        self.moveCost_penalized = 10 ** 9


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


def calculate_cost_matrix(sequence_of_nodes,rows):
    """
    :param sequence_of_nodes: sequence of customer Nodes
    :return: a List of List objects containing the cost from one node to another for each node
    """
    matrix = [[0 for _ in range(rows)] for _ in range(rows)]
    for from_index in range(len(sequence_of_nodes)):
        for to_index in range(from_index + 1, len(sequence_of_nodes)):
            from_n = sequence_of_nodes[from_index]
            to_n = sequence_of_nodes[to_index]
            cost = calculate_euclid(from_n, to_n)
            matrix[from_index][to_index] = cost
            matrix[to_index][from_index] = cost
    return matrix


def generate_initial_population(population_size, sequence_of_nodes, matrix, initial_node, capacity, vehicles):
    """
    :param population_size: how big is the population
    :param sequence_of_nodes: List of Node objects
    :param matrix: the distance-cost matrix
    :param initial_node: depot to be the start of each route
    :param capacity: how much load can a route hold
    :param vehicles: the number of vehicles
    :return: a List containing a population of Genomes
    Using the population size, a list of genomes is created by assigning each node to a vehicle and order.
    After each genome is created, fitness score is calculated and then is appended to the list.
    After every genome is created the population gets sorted in ascending order.
    """
    pop = []
    # Initialize the population with random genomes
    for _ in range(population_size):
        new_genome = Genome()
        for _ in range(len(sequence_of_nodes)):
            in_route_position = random()
            assigned_vehicle = randint(1, vehicles)
            new_genome.append(in_route_position + assigned_vehicle)
        # calculate the genome's fitness score
        new_genome.fitness(sequence_of_nodes, matrix, initial_node, capacity, vehicles)
        pop.append(new_genome)

    pop.sort(key=lambda x: x.score)  # sort by the fitness score

    return pop


def manage_duplicates(pop, mutation_rate, number_of_vehicles, sequence_of_nodes, matrix, initial_node, capacity):
    """
    :param pop: population of genomes
    :param mutation_rate: rate in which a single gene is mutated
    :param number_of_vehicles: how many vehicles in VRP
    :param sequence_of_nodes: a List of Node objects
    :param matrix: distance-cost matrix
    :param initial_node: depot, the starting Node of each route
    :param capacity: maximum load of vehicles
    :return: None
    For every genome in the population check if there are duplicated genomes. If that's the case, then mutate those
    genomes until they are different.
    """
    unique_genomes = set()
    for pos, genome in enumerate(pop):
        while tuple(genome) in unique_genomes:
            genome.mutate(mutation_rate, number_of_vehicles)
            # calculate new score of genome
            genome.fitness(sequence_of_nodes, matrix, initial_node, capacity, number_of_vehicles)
        pop[pos] = genome
        unique_genomes.add(tuple(genome))

def genetic_algorithm(population_size, sequence_of_nodes, matrix, initial_node, capacity, vehicles, generations,
                      mutation_rate, gene_selection_rate):
    """
    :param population_size: how big the population is
    :param sequence_of_nodes: a List of Node objects
    :param matrix: distance-cost matrix
    :param initial_node: depot Node
    :param capacity: how much load can vehicles handle
    :param vehicles: number of vehicles
    :param generations: the number of iterations to use genetic algorithm
    :param mutation_rate: the chance of a gene mutating
    :param gene_selection_rate: chance of first genome's gene being selected
    :return: the final population of genomes
    Implementation of Genetic Algorithm, using elitism logic. Constructs initial population, and then for a number of
    generations: crossovers genomes with an elite genome(top 5% of the population), and chooses a random portion to
    mutate. After mutating the population, duplicate genomes are found and mutated yet another time.
    """
    pop = generate_initial_population(population_size, sequence_of_nodes, matrix, initial_node, capacity, vehicles)

    number_of_elites = int(0.05 * population_size)
    chromosomes_to_mutate = int(0.2 * population_size)
    non_elite_genomes = list(range(number_of_elites, len(pop)))
    # lesser_elites = list(range(number_of_elites // 2, number_of_elites))
    elites = list(range(number_of_elites))

    for generation in range(generations):
        new_population = pop[:number_of_elites]
        shuffle(non_elite_genomes)
        for genome in non_elite_genomes:
            cross_genome = choice(elites)
            new_population.append(pop[genome])
            new_population[-1].crossover(new_population[cross_genome], gene_selection_rate)
            new_population[-1].fitness(sequence_of_nodes, matrix, initial_node, capacity, vehicles)
        for i in range(chromosomes_to_mutate):
            genome = randrange(len(new_population))
            new_population[genome].mutate(mutation_rate, vehicles)
            new_population[genome].fitness(sequence_of_nodes, matrix, initial_node, capacity, vehicles)
        manage_duplicates(new_population, mutation_rate, vehicles, sequence_of_nodes, matrix, initial_node, capacity)
        new_population.sort(key=lambda x: x.score)
        pop = new_population
        print(f'Generation: {generation}', pop[0].score)
    return pop

def clone_route(rt: Route, depot, capacity):
    cloned = Route(depot, capacity)
    cloned.cumulative_cost = rt.cumulative_cost
    cloned.load = rt.load
    cloned.sequenceOfNodes = rt.sequenceOfNodes.copy()
    return cloned

#for relocations- ton xreisimopoiei h eygenia 
def calculate_sol_cumulative(sol: Solution, matrix):
    totalSolCost = 0
    for r in range(0, len(sol.routes)):
        rt: Route = sol.routes[r]
        rtCost = 0
        rtCumCost = 0
        for n in range(len(rt.sequenceOfNodes) - 2): #an exeis vgalei to dummy otan eisai edw allakse tous deiktes
            A = rt.sequenceOfNodes[n]
            B = rt.sequenceOfNodes[n + 1]
            rtCost += matrix[A.ID][B.ID]
            rtCumCost += rtCost
            rtCost += B.unloading_time
        totalSolCost += rtCumCost
    return totalSolCost

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

def calculate_pen_cum_matrix(route: Route,pen: Penalize):
    rt_cumulative_cost = 0
    tot_time = 0
    
    route.pen_cum_matrix = []
    for i in range(len(route.sequenceOfNodes) - 1):
        print(i)
        from_node: Node = route.sequenceOfNodes[i]
        to_node: Node = route.sequenceOfNodes[i + 1]
        tot_time += pen.distance_matrix_penalized[from_node.ID][to_node.ID]
        rt_cumulative_cost += tot_time
        route.pen_cum_matrix.append(rt_cumulative_cost)
        tot_time += to_node.unloading_time

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

    cost_added = calculate_route_cumulative(node_sequence_route_1, matrix) \
        + calculate_route_cumulative(node_sequence_route_2, matrix)
    return cost_added - cost_removed



def FindBestRelocationMove(rm, sol: Solution, matrix, pen:Penalize):
    dummy = Node(101,0,0,0,0)
    for originRouteIndex in range(0, len(sol.routes)):
            rt1: Route = sol.routes[originRouteIndex]
            rt1.sequenceOfNodes.append(dummy)
            for originNodeIndex in range(1, len(rt1.sequenceOfNodes) - 1):
                for targetRouteIndex in range(0, len(sol.routes)):
                    rt2: Route = sol.routes[targetRouteIndex]
                    rt2.sequenceOfNodes.append(dummy)
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
                        print(originNodeIndex)
                        costRemoved_penalized = (pen.distance_matrix_penalized[A.ID][B.ID] + B.unloading_time + \
                                                 pen.distance_matrix_penalized[B.ID][C.ID])*(len1-indexC) + \
                                                 pen.distance_matrix_penalized[F.ID][G.ID]*(len2-indexG) + \
                                                 rt1.pen_cum_matrix[originNodeIndex]
                        

                        costAdded_penalized = pen.distance_matrix_penalized[A.ID][C.ID]*(len1 - indexC) +\
                                    (pen.distance_matrix_penalized[F.ID][B.ID] + B.unloading_time + pen.distance_matrix_penalized[B.ID][G.ID])*(len2-indexG) +\
                                    + rt2.pen_cum_matrix[targetNodeIndex] + F.unloading_time + pen.distance_matrix_penalized[F.ID][B.ID]

                        moveCost_penalized = costAdded_penalized - costRemoved_penalized

                        if (moveCost_penalized < rm.moveCost_penalized):
                            StoreBestRelocationMove(originRouteIndex, targetRouteIndex, originNodeIndex,
                                                         targetNodeIndex, moveCost_penalized, rm)
                    rt2.sequenceOfNodes.pop()
            rt1.sequenceOfNodes.pop()
            
def FindBestTwoOptMove(top, sol: Solution, matrix, pen : Penalize):
    for index_of_route_1 in range(0, len(sol.routes)):
        route_1: Route = sol.routes[index_of_route_1]
        for index_of_route_2 in range(index_of_route_1, len(sol.routes)):
            route_2: Route = sol.routes[index_of_route_2]
            for index_of_node_1 in range(0, len(route_1.sequenceOfNodes) - 1):
                start2 = 0
                if route_1 == route_2:
                    start2 = index_of_node_1 + 2
                for index_of_node_2 in range(start2, len(route_2.sequenceOfNodes) - 1):
                    
                    A = route_1.sequenceOfNodes[index_of_node_1]
                    B = route_1.sequenceOfNodes[index_of_node_1 + 1]
                    K = route_2.sequenceOfNodes[index_of_node_2]
                    L = route_2.sequenceOfNodes[index_of_node_2 + 1]

                    if route_1 == route_2:
                        if index_of_node_1 == 0 and index_of_node_2 == len(route_1.sequenceOfNodes) - 2:
                            continue
                        move_cost = calculate_intra_route_cumulative_change(index_of_node_1, index_of_node_2, route_1, matrix)
                        
                        costAdded_penalized = pen.distance_matrix_penalized[A.ID][K.ID] + pen.distance_matrix_penalized[B.ID][L.ID]
                        costRemoved_penalized = pen.distance_matrix_penalized[A.ID][B.ID] + pen.distance_matrix_penalized[K.ID][L.ID]
                        moveCost_penalized = costAdded_penalized - costRemoved_penalized
                    else:
                        if index_of_node_1 == 0 and index_of_node_2 == 0:
                            continue
                        if index_of_node_1 == len(route_1.sequenceOfNodes) - 2 and index_of_node_2 == len(
                                route_2.sequenceOfNodes) - 2:
                            continue

                        if capacity_is_violated(route_1, index_of_node_1, route_2, index_of_node_2):
                            continue
                        move_cost = calculate_inter_route_cumulative_change(index_of_node_1, index_of_node_2, route_1,
                                                                            route_2, matrix)
                        costAdded_penalized = pen.distance_matrix_penalized[A.ID][L.ID] + pen.distance_matrix_penalized[B.ID][K.ID]
                        costRemoved_penalized = pen.distance_matrix_penalized[A.ID][B.ID] + pen.distance_matrix_penalized[K.ID][L.ID]
                        moveCost_penalized = costAdded_penalized - costRemoved_penalized
                    if moveCost_penalized < top.moveCost_penalized:
                        StoreBestTwoOptMove(index_of_route_1, index_of_route_2, index_of_node_1, index_of_node_2,
                                                move_cost,moveCost_penalized, top)
                
def FindBestSwapMove(sm, sol: Solution, matrix, pen:Penalize):
        for firstRouteIndex in range(0, len(sol.routes)):
            rt1:Route = sol.routes[firstRouteIndex]
            for secondRouteIndex in range (firstRouteIndex, len(sol.routes)):
                rt2:Route = sol.routes[secondRouteIndex]
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
                                costRemoved = matrix[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + matrix[b1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1) + \
                                              matrix[b2.ID][c2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-2)
                                costAdded = matrix[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + matrix[b2.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1) + \
                                            matrix[b1.ID][c2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-2)
                                moveCost = costAdded - costRemoved

                                costRemoved_penalized = pen.distance_matrix_penalized[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + pen.distance_matrix_penalized[b1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1) + \
                                                        pen.distance_matrix_penalized[b2.ID][c2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-2)
                                costAdded_penalized = pen.distance_matrix_penalized[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + pen.distance_matrix_penalized[b2.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1) + \
                                                      pen.distance_matrix_penalized[b1.ID][c2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-2)
                                moveCost_penalized = costAdded_penalized - costRemoved_penalized
                            else:

                                costRemoved1 = matrix[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + matrix[b1.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                                costAdded1 = matrix[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + matrix[b2.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                                costRemoved2 = matrix[a2.ID][b2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + matrix[b2.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                                costAdded2 = matrix[a2.ID][b1.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + matrix[b1.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                                moveCost = costAdded1 + costAdded2 - (costRemoved1 + costRemoved2)
                                
                                costRemoved1_penalized = pen.distance_matrix_penalized[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + pen.distance_matrix_penalized[b1.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                                costAdded1_penalized = pen.distance_matrix_penalized[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + pen.distance_matrix_penalized[b2.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                                costRemoved2_penalized = pen.distance_matrix_penalized[a2.ID][b2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + pen.distance_matrix_penalized[b2.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                                costAdded2_penalized = pen.distance_matrix_penalized[a2.ID][b1.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + matrix[b1.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                                moveCost_penalized = costAdded1_penalized + costAdded2_penalized - (costRemoved1_penalized + costRemoved2_penalized)
                        else:
                            if rt1.load - b1.demand + b2.demand > rt1.capacity:
                                continue
                            if rt2.load - b2.demand + b1.demand > rt2.capacity:
                                continue

                            costRemoved1 = matrix[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + matrix[b1.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                            costAdded1 = matrix[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + matrix[b2.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                            costRemoved2 = matrix[a2.ID][b2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + matrix[b2.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                            costAdded2 = matrix[a2.ID][b1.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + matrix[b1.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                            
                            costRemoved1_penalized = pen.distance_matrix_penalized[a1.ID][b1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + pen.distance_matrix_penalized[b1.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                            costAdded1_penalized = pen.distance_matrix_penalized[a1.ID][b2.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex) + pen.distance_matrix_penalized[b2.ID][c1.ID]*(len(rt1.sequenceOfNodes)-firstNodeIndex-1)
                            costRemoved2_penalized = pen.distance_matrix_penalized[a2.ID][b2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + pen.distance_matrix_penalized[b2.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)
                            costAdded2_penalized = pen.distance_matrix_penalized[a2.ID][b1.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex) + matrix[b1.ID][c2.ID]*(len(rt2.sequenceOfNodes)-secondNodeIndex-1)

                            costChangeFirstRoute = costAdded1 - costRemoved1
                            costChangeSecondRoute = costAdded2 - costRemoved2

                            moveCost = costAdded1 + costAdded2 - (costRemoved1 + costRemoved2)
                            moveCost_penalized = costAdded1_penalized + costAdded2_penalized - (
                                        costRemoved1_penalized + costRemoved2_penalized)

                        if moveCost_penalized < sm.moveCost_penalized:
                            StoreBestSwapMove(firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex,
                                                   moveCost, moveCost_penalized, costChangeFirstRoute, costChangeSecondRoute, sm)

def capacity_is_violated(route_1, index_of_node_1, route_2, index_of_node_2):
    load_of_route_1_first_segment = sum(route_1.sequenceOfNodes[i].demand for i in range(0, index_of_node_1 + 1))
    load_of_route_1_second_segment = route_1.load - load_of_route_1_first_segment

    load_of_route_2_first_segment = sum(route_2.sequenceOfNodes[i].demand for i in range(0, index_of_node_2 + 1))
    load_of_route_2_second_segment = route_2.load - load_of_route_2_first_segment

    route_1_violated = load_of_route_1_first_segment + load_of_route_2_second_segment > route_1.capacity
    route_2_violated = load_of_route_2_first_segment + load_of_route_1_second_segment > route_2.capacity

    return route_1_violated or route_2_violated

def StoreBestRelocationMove( originRouteIndex, targetRouteIndex, originNodeIndex, targetNodeIndex,
                            moveCost_penalized, rm: RelocationMove):
    rm.originRoutePosition = originRouteIndex
    rm.originNodePosition = originNodeIndex
    rm.targetRoutePosition = targetRouteIndex
    rm.targetNodePosition = targetNodeIndex
    rm.moveCost_penalized = moveCost_penalized

def StoreBestTwoOptMove(index_of_route_1, index_of_route_2, index_of_node_1, index_of_node_2, move_cost, moveCost_penalized, top):
    top.positionOfFirstRoute = index_of_route_1
    top.positionOfSecondRoute = index_of_route_2
    top.positionOfFirstNode = index_of_node_1
    top.positionOfSecondNode = index_of_node_2
    top.moveCost = move_cost
    top.moveCost_penalized = moveCost_penalized

def StoreBestSwapMove(firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex, moveCost,
                          moveCost_penalized, costChangeFirstRoute, costChangeSecondRoute, sm):
        sm.positionOfFirstRoute = firstRouteIndex
        sm.positionOfSecondRoute = secondRouteIndex
        sm.positionOfFirstNode = firstNodeIndex
        sm.positionOfSecondNode = secondNodeIndex
        sm.costChangeFirstRt = costChangeFirstRoute
        sm.costChangeSecondRt = costChangeSecondRoute
        sm.moveCost = moveCost
        sm.moveCost_penalized = moveCost_penalized

def ApplyTwoOptMove(top, sol: Solution, matrix):
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

        route_1.cumulative_cost += top.moveCost

    else:
        # slice with the nodes from position top.positionOfFirstNode + 1 onwards
        relocated_segment_of_route_1 = route_1.sequenceOfNodes[top.positionOfFirstNode + 1:]

        # slice with the nodes from position top.positionOfFirstNode + 1 onwards
        relocated_segment_of_route_2 = route_2.sequenceOfNodes[top.positionOfSecondNode + 1:]

        del route_1.sequenceOfNodes[top.positionOfFirstNode + 1:]
        del route_2.sequenceOfNodes[top.positionOfSecondNode + 1:]

        route_1.sequenceOfNodes.extend(relocated_segment_of_route_2)
        route_2.sequenceOfNodes.extend(relocated_segment_of_route_1)

        update_route_cost_and_load(route_1, matrix)
        update_route_cost_and_load(route_2, matrix)

    sol.total_cumulative_cost += top.moveCost
    
def ApplySwapMove(sm, sol):
        oldCost = sol.total_cumulative_cost
        rt1 = sol.routes[sm.positionOfFirstRoute]
        rt2 = sol.routes[sm.positionOfSecondRoute]
        b1 = rt1.sequenceOfNodes[sm.positionOfFirstNode]
        b2 = rt2.sequenceOfNodes[sm.positionOfSecondNode]
        rt1.sequenceOfNodes[sm.positionOfFirstNode] = b2
        rt2.sequenceOfNodes[sm.positionOfSecondNode] = b1

        if (rt1 == rt2):
            rt1.cumulative_cost += sm.moveCost
        else:
            rt1.cumulative_cost += sm.costChangeFirstRt
            rt2.cumulative_cost += sm.costChangeSecondRt
            rt1.load = rt1.load - b1.demand + b2.demand
            rt2.load = rt2.load + b1.demand - b2.demand

        sol.total_cumulative_cost += sm.moveCost

        newCost = sol.total_cumulative_cost
        # debuggingOnly
        if abs((newCost - oldCost) - sm.moveCost) > 0.0001:
            print('Cost Issue')

def ApplyRelocationMove(sol: Solution, rm: RelocationMove, matrix,pen:Penalize):

    originRt = sol.routes[rm.originRoutePosition]
    targetRt = sol.routes[rm.targetRoutePosition]

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

    calculate_pen_cum_matrix(originRt,pen)
    calculate_pen_cum_matrix(targetRt,pen)
    sol.total_cumulative_cost = calculate_sol_cumulative(sol,matrix)


def update_route_cost_and_load(route: Route, matrix):
    route.cumulative_cost = calculate_route_cumulative(route.sequenceOfNodes, matrix)
    route.load = sum(node.demand for node in route.sequenceOfNodes)

def penalize_arcs(sol : Solution, matrix, pen : Penalize):
        # if pen.penalized_n1_ID != -1 and pen.penalized_n2_ID != -1:
        #     pen.distance_matrix_penalized[pen.penalized_n1_ID][pen.penalized_n2_ID] = matrix[pen.penalized_n1_ID][pen.penalized_n2_ID]
        #     pen.distance_matrix_penalized[pen.penalized_n2_ID][pen.penalized_n1_ID] = matrix[pen.penalized_n2_ID][pen.penalized_n1_ID]
        max_criterion = 0
        pen_1 = -1
        pen_2 = -1
        for i in range(len(sol.routes)):
            rt = sol.routes[i]
            for j in range(len(rt.sequenceOfNodes) - 1):
                id1 = rt.sequenceOfNodes[j].ID
                id2 = rt.sequenceOfNodes[j + 1].ID
                criterion = matrix[id1][id2] / (1 + pen.times_penalized[id1][id2])
                if criterion > max_criterion:
                    max_criterion = criterion
                    pen_1 = id1
                    pen_2 = id2
        pen.times_penalized[pen_1][pen_2] += 1
        pen.times_penalized[pen_2][pen_1] += 1

        pen_weight = 0.15

        pen.distance_matrix_penalized[pen_1][pen_2] = (1 + pen_weight * pen.times_penalized[pen_1][pen_2]) * matrix[pen_1][pen_2]
        pen.distance_matrix_penalized[pen_2][pen_1] = (1 + pen_weight * pen.times_penalized[pen_2][pen_1]) * matrix[pen_2][pen_1]
        pen.penalized_n1_ID = pen_1
        pen.penalized_n2_ID = pen_2

def calculate_initial_best_solution(population, nodes, depot, capacity, number_of_vehicles, distance_matrix):
    best = None
    for final_genome in population:
        best = final_genome
        if best.feasible:
            break

    routes = best.create_route_list(nodes, depot, capacity, number_of_vehicles, distance_matrix)
    return Solution(routes)

#-----------------------------------------------------------------------------------------------------------

def local_search(initial_solution, depot, capacity, matrix, pen: Penalize, number_of_iterations):
    
    solution = initial_solution.clone_solution(depot, capacity)
    best_solution = initial_solution.clone_solution(depot, capacity)

    terminationCondition = False
    localSearchIterator = 0

    rm = RelocationMove()
    sm = SwapMove()
    top = TwoOptMove()

    while terminationCondition is False:
            
            operator = randint(1,2)
            
            top.Initialize()
            sm.Initialize()
            #rm.Initialize()
           
            # Relocations-UnderConstruction
            # if operator == 0:
            #     FindBestRelocationMove(rm,solution,matrix,pen)
            #     if rm.originRoutePosition is not None:
            #         if rm.moveCost_penalized < 0:
            #             ApplyRelocationMove(solution,rm,matrix,pen)
            #         else:
            #             penalize_arcs(solution,matrix,pen)
            #             localSearchIterator = localSearchIterator - 1
            
            # Swaps
            if operator == 1:
                FindBestSwapMove(sm,solution,matrix,pen)
                if sm.positionOfFirstRoute is not None:
                    if sm.moveCost_penalized < 0:
                        ApplySwapMove(sm,solution)
                    else:
                        penalize_arcs(solution,matrix,pen)
                        localSearchIterator = localSearchIterator - 1
                        
            # TwoOpt
            elif operator == 2:
                FindBestTwoOptMove(top,solution,matrix,pen)
                if top.positionOfFirstRoute is not None:
                    if top.moveCost_penalized < 0:
                        ApplyTwoOptMove(top,solution,matrix)
                        
                    else:
                        penalize_arcs(solution,matrix,pen)
                        localSearchIterator = localSearchIterator - 1
            

            if (solution.total_cumulative_cost < best_solution.total_cumulative_cost):
                best_solution = solution.clone_solution(depot, capacity)
                print(f'Iteration: {localSearchIterator}, Solution cost: {solution.total_cumulative_cost},'
              f' best Solution cost: {best_solution.total_cumulative_cost}'
              f' Move:  {operator}')

            localSearchIterator = localSearchIterator + 1
            if localSearchIterator == number_of_iterations:
                terminationCondition = True

    return best_solution


#-----------------------------------------------------------------------------------------------------------
 
def main():
    start_time = time.time()

    number_of_vehicles, capacity, customers, nodes, depot = read_vrp_problem_info()
    distance_matrix = calculate_cost_matrix([depot] + nodes,len([depot] + nodes))
    rows = len(nodes) + 1
    pen = Penalize(rows,distance_matrix)

    seed(5)
    population_size = 1000
    
    generations = 700
    mutation_rate = 0.5
    gene_selection_rate = 0.3

    population = genetic_algorithm(population_size, nodes, distance_matrix, depot, capacity, number_of_vehicles,
                                   generations, mutation_rate, gene_selection_rate)

    initial_best_solution = calculate_initial_best_solution(population, nodes, depot, capacity,
                                                            number_of_vehicles, distance_matrix)
    
    number_of_iterations = 5000
    
    best_solution = local_search(initial_best_solution, depot, capacity, distance_matrix, pen, number_of_iterations)
    
    write_solution_to_file(best_solution, number_of_vehicles)
    end_time = time.time()
    print(f'time elapsed: {end_time - start_time}')


if __name__ == '__main__':
    main()
