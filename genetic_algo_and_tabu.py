# https://trace.tennessee.edu/cgi/viewcontent.cgi?article=6358&context=utk_gradthes#:~:text=The%20genetic%20algorithm%20was%20developed,routing%20problems%2C%20and%20many%20others.

import time
from math import sqrt
from random import randint, randrange, seed, random, shuffle, choice


class Node:
    def __init__(self, id_n=0, x_cor=0, y_cor=0, dem=0, un_time=0):
        self.id = id_n
        self.x = x_cor
        self.y = y_cor
        self.demand = dem
        self.unloading_time = un_time
        self.tabu_until = -1

    def from_line(self, line, sep=','):
        split_line = line.split(sep)
        self.id = int(split_line[0])
        self.x = int(split_line[1])
        self.y = int(split_line[2])
        self.demand = int(split_line[3])
        self.unloading_time = int(split_line[4])

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f'Node({self.id}, {self.x}, {self.y}, {self.demand}, {self.unloading_time})'

    def __repr__(self):
        return str(self)


class Route:
    def __init__(self, initial_node, max_capacity):
        self.nodes = [initial_node]
        self.total = 0
        self.cum = 0
        self.load = 0
        self.MAX_LOAD = max_capacity

    def add_node(self, new_node: Node, minimum_cost):
        self.nodes.append(new_node)
        self.total += minimum_cost
        self.cum += self.total
        self.total += new_node.unloading_time
        self.load += new_node.demand

    def last(self):
        return self.nodes[-1]

    def __str__(self):
        return ' - '.join(list(str(node.id) for node in self.nodes))

    def __repr__(self):
        return str(self)


class Solution(list):
    def __init__(self):
        super().__init__()
        self.cum = 0
        self.total = 0
        self.score = 0
        self.feasible = True

    def fitness(self, nodes, matrix, depot, capacity, number_of_vehicles):
        self.feasible = True
        node_sorted = [(node, random_key) for node, random_key in zip(nodes, self)]
        node_sorted += [(depot, vehicle + 1) for vehicle in range(number_of_vehicles)]
        node_sorted.sort(key=lambda x: x[1])
        cum = 0
        non_feasible_vehicles = 0
        prev = None
        for node in node_sorted:
            if node[0] is depot:
                route_total = 0
                route_cap = 0
                unchanged_feasibility = True
            else:
                weight = matrix[node[0].id][prev[0].id]
                route_total += weight
                self.total += weight
                cum += route_total
                route_total += node[0].unloading_time
                route_cap += node[0].demand
                if route_cap > capacity and unchanged_feasibility:
                    unchanged_feasibility = False
                    non_feasible_vehicles += 1
            prev = node
        self.cum = cum
        if non_feasible_vehicles > 0:
            self.feasible = False
            self.score = self.cum + number_of_vehicles * (self.total - number_of_vehicles * capacity) ** 2
        else:
            self.score = self.cum

    def crossover(self, gene, gene_selection_rate):
        for i in range(len(self)):
            selection = random()
            if selection >= gene_selection_rate:
                self[i] = gene[i]

    def mutate(self, mutation_rate, number_of_vehicles):
        for i, val in enumerate(self):
            mutation = random()
            if mutation < mutation_rate:
                # val - int(val)
                decimal = random()
                new_vehicle = randint(1, number_of_vehicles)
                self[i] = new_vehicle + decimal

    def create_route_list(self, nodes, depot: Node, capacity):
        routes = []
        node_sorted = [(node, value) for node, value in zip(nodes, self)]
        node_sorted += [(depot, i + 1) for i in range(VEHICLE_NUMBER)]
        node_sorted.sort(key=lambda x: x[1])
        for node, _ in node_sorted:
            if node == depot:
                routes.append(Route(depot, capacity))
            else:
                cost = cost_matrix[routes[-1].last().id][node.id]
                routes[-1].add_node(node, cost)
        return routes


class TwoOptSolution:
    def __init__(self, routes):
        self.routes = routes
        self.total_cum_cost = sum(r.cum for r in routes)


class TwoOptMove:
    def __init__(self):
        self.position_of_first_route = None
        self.position_of_second_route = None
        self.position_of_first_node = None
        self.position_of_second_node = None
        self.move_total_cost = None

    def initialize(self):
        self.position_of_first_route = None
        self.position_of_second_route = None
        self.position_of_first_node = None
        self.position_of_second_node = None
        self.move_total_cost = 10 ** 9


class Tabu:
    def __init__(self, minimum, maximum):
        self.min_duration = minimum
        self.max_duration = maximum

    def move_is_tabu(self, current_solution: TwoOptSolution, best_solution: TwoOptSolution, node: Node, move_cost, iterator, epsilon=0.001):
        if current_solution.total_cum_cost + move_cost < best_solution.total_cum_cost - epsilon:
            return False
        return iterator < node.tabu_until

    def set_tabu(self, tabu_node: Node, iterator):
        tabu_node.tabu_until = iterator + randint(self.min_duration, self.max_duration)


def get_int_from_line(line, pos=1, sep=','):
    return int(line.split(sep)[pos])


def calculate_euclid(node_from, node_to):
    return sqrt((node_from.x - node_to.x) ** 2 + (node_from.y - node_to.y) ** 2)


def calculate_cost_matrix(nodes):
    matrix = [[0 for j in range(len(nodes))] for i in range(len(nodes))]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            from_n = nodes[i]
            to_n = nodes[j]
            cost = calculate_euclid(from_n, to_n)
            matrix[i][j] = cost
            matrix[j][i] = cost
    return matrix


def manage_duplicates(pop):
    unique_genes = set()
    for pos, gene in enumerate(pop):
        while tuple(gene) in unique_genes:
            gene.mutate(MUT_RATE, VEHICLE_NUMBER)
            gene.fitness(nodes, cost_matrix, depot, CAPACITY, VEHICLE_NUMBER)
        pop[pos] = gene
        unique_genes.add(tuple(gene))


########################## Two Opt Functions ##########################


def clone_route(rt: Route):
    cloned = Route(depot, CAPACITY)
    cloned.cum = rt.cum
    cloned.load = rt.load
    cloned.nodes = rt.nodes.copy()
    return cloned


def clone_solution(sol: TwoOptSolution):
    cloned_routes = []
    for i in range(0, len(sol.routes)):
        rt = sol.routes[i]
        cloned_route = clone_route(rt)
        cloned_routes.append(cloned_route)
    cloned = TwoOptSolution(cloned_routes)
    cloned.total_cum_cost = sol.total_cum_cost
    return cloned


def calculate_route_cumulative(nodes_sequence):
    rt_cumulative_cost = 0
    tot_time = 0
    for i in range(len(nodes_sequence) - 1):
        from_node: Node = nodes_sequence[i]
        to_node: Node = nodes_sequence[i + 1]
        tot_time += cost_matrix[from_node.id][to_node.id]
        rt_cumulative_cost += tot_time
        tot_time += to_node.unloading_time
    return rt_cumulative_cost


def calculate_intra_route_cumulative_change(index_of_node_1, index_of_node_2, route):
    cost_removed = route.cum

    node_sequence_copy = route.nodes[:]
    reversed_segment = reversed(node_sequence_copy[index_of_node_1 + 1: index_of_node_2 + 1])
    node_sequence_copy[index_of_node_1 + 1: index_of_node_2 + 1] = reversed_segment

    cost_added = calculate_route_cumulative(node_sequence_copy)
    return cost_added - cost_removed


def calculate_inter_route_cumulative_change(index_of_node_1, index_of_node_2, route_1, route_2):
    cost_removed = route_1.cum + route_2.cum

    node_sequence_route_1 = route_1.nodes[:]
    # slice with the nodes from position top.positionOfFirstNode + 1 onwards
    relocated_segment_of_route_1 = node_sequence_route_1[index_of_node_1 + 1:]

    node_sequence_route_2 = route_2.nodes[:]
    # slice with the nodes from position top.positionOfFirstNode + 1 onwards
    relocated_segment_of_route_2 = node_sequence_route_2[index_of_node_2 + 1:]

    del node_sequence_route_1[index_of_node_1 + 1:]
    del node_sequence_route_2[index_of_node_2 + 1:]

    node_sequence_route_1.extend(relocated_segment_of_route_2)
    node_sequence_route_2.extend(relocated_segment_of_route_1)

    cost_added = calculate_route_cumulative(node_sequence_route_1) + calculate_route_cumulative(node_sequence_route_2)
    return cost_added - cost_removed


def find_best_two_opt_move(top, sol: TwoOptSolution, best_sol: TwoOptSolution, tabu_object: Tabu, iteration):
    for index_of_route_1 in range(0, len(sol.routes)):
        route_1: Route = sol.routes[index_of_route_1]
        for index_of_route_2 in range(index_of_route_1, len(sol.routes)):
            route_2: Route = sol.routes[index_of_route_2]
            for index_of_node_1 in range(0, len(route_1.nodes) - 1):
                start2 = 0
                if route_1 == route_2:
                    start2 = index_of_node_1 + 2
                for index_of_node_2 in range(start2, len(route_2.nodes) - 1):

                    if route_1 == route_2:
                        if index_of_node_1 == 0 and index_of_node_2 == len(route_1.nodes) - 2:
                            continue
                        move_cost = calculate_intra_route_cumulative_change(index_of_node_1, index_of_node_2, route_1)
                    else:
                        if index_of_node_1 == 0 and index_of_node_2 == 0:
                            continue
                        if index_of_node_1 == len(route_1.nodes) - 2 and index_of_node_2 == len(route_2.nodes) - 2:
                            continue

                        if capacity_is_violated(route_1, index_of_node_1, route_2, index_of_node_2):
                            continue
                        move_cost = calculate_inter_route_cumulative_change(index_of_node_1, index_of_node_2, route_1,
                                                                                    route_2)

                    if tabu_object.move_is_tabu(sol, best_sol, route_1.nodes[index_of_node_1], move_cost, iteration) or \
                            tabu_object.move_is_tabu(sol, best_sol, route_2.nodes[index_of_node_2], move_cost, iteration):
                        continue

                    if move_cost < top.move_total_cost:
                        store_best_two_opt_move(index_of_route_1, index_of_route_2, index_of_node_1, index_of_node_2,
                                                move_cost, top)


def capacity_is_violated(route_1, index_of_node_1, route_2, index_of_node_2):
    load_of_route_1_first_segment = sum(route_1.nodes[i].demand for i in range(0, index_of_node_1 + 1))
    load_of_route_1_second_segment = route_1.load - load_of_route_1_first_segment

    load_of_route_2_first_segment = sum(route_2.nodes[i].demand for i in range(0, index_of_node_2 + 1))
    load_of_route_2_second_segment = route_2.load - load_of_route_2_first_segment

    route_1_violated = load_of_route_1_first_segment + load_of_route_2_second_segment > route_1.MAX_LOAD
    route_2_violated = load_of_route_2_first_segment + load_of_route_1_second_segment > route_2.MAX_LOAD

    return route_1_violated or route_2_violated


def store_best_two_opt_move(index_of_route_1, index_of_route_2, index_of_node_1, index_of_node_2, move_cost, top):
    top.position_of_first_route = index_of_route_1
    top.position_of_second_route = index_of_route_2
    top.position_of_first_node = index_of_node_1
    top.position_of_second_node = index_of_node_2
    top.move_total_cost = move_cost


def apply_two_opt_move(top, sol: TwoOptSolution, tabu: Tabu, iteration):
    route_1: Route = sol.routes[top.position_of_first_route]
    route_2: Route = sol.routes[top.position_of_second_route]
    node_1 = route_1.nodes[top.position_of_first_node]
    node_2 = route_2.nodes[top.position_of_second_node]

    if route_1 == route_2:
        # reverses the nodes in the segment [positionOfFirstNode + 1,  top.positionOfSecondNode]
        reversed_segment = reversed(route_1.nodes[top.position_of_first_node + 1: top.position_of_second_node + 1])
        # lst = list(reversed_segment)
        # lst2 = list(reversed_segment)
        route_1.nodes[top.position_of_first_node + 1: top.position_of_second_node + 1] = reversed_segment

        # reversedSegmentList = list(reversed(route_1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1]))
        # route_1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegmentList

        route_1.cum += top.move_total_cost

    else:
        # slice with the nodes from position top.positionOfFirstNode + 1 onwards
        relocated_segment_of_route_1 = route_1.nodes[top.position_of_first_node + 1:]

        # slice with the nodes from position top.positionOfFirstNode + 1 onwards
        relocated_segment_of_route_2 = route_2.nodes[top.position_of_second_node + 1:]

        del route_1.nodes[top.position_of_first_node + 1:]
        del route_2.nodes[top.position_of_second_node + 1:]

        route_1.nodes.extend(relocated_segment_of_route_2)
        route_2.nodes.extend(relocated_segment_of_route_1)

        update_route_cost_and_load(route_1)
        update_route_cost_and_load(route_2)

    sol.total_cum_cost += top.move_total_cost
    tabu.set_tabu(node_1, iteration)
    tabu.set_tabu(node_2, iteration)


def update_route_cost_and_load(route: Route):
    route.cum = calculate_route_cumulative(route.nodes)
    route.load = sum(node.demand for node in route.nodes)


####################################################################################################################


start_time = time.time()

with open('../Instance.txt') as f:
    VEHICLE_NUMBER = get_int_from_line(f.readline())
    CAPACITY = get_int_from_line(f.readline())
    CUSTOMERS = get_int_from_line(f.readline())
    f.readline()  # get rid of NODE INFO
    f.readline()  # get rid of column names
    depot = Node()
    depot.from_line(f.readline())
    nodes = []
    for line in f.readlines():
        node = Node()
        node.from_line(line)
        nodes.append(node)

seed(1)
POPULATION_SIZE = 2000
population = []
for i in range(POPULATION_SIZE):
    solution = Solution()
    for _ in range(len(nodes)):
        rand = random()
        random_int = randint(1, VEHICLE_NUMBER)
        solution.append(rand + random_int)
    population.append(solution)

cost_matrix = calculate_cost_matrix([depot] + nodes)
for sol in population:
    sol.fitness(nodes, cost_matrix, depot, CAPACITY, VEHICLE_NUMBER)

population.sort(key=lambda x: x.score)

GENERATIONS = 500

MUT_RATE = 0.5
GEN_RATE = 0.3
NUMBER_OF_ELITES = int(0.05 * POPULATION_SIZE)
CHROMOSOMES_TO_MUTATE = int(0.2 * POPULATION_SIZE)
sols = list(range(NUMBER_OF_ELITES, len(population)))
# lesser_elites = list(range(NUMBER_OF_ELITES // 2, NUMBER_OF_ELITES))
elites = list(range(NUMBER_OF_ELITES))
best_solution_per_generation = []

for generation in range(GENERATIONS):
    new_population = population[:NUMBER_OF_ELITES]
    shuffle(sols)
    for gene in sols:
        cross_gene = choice(elites)
        new_population.append(population[gene])
        new_population[-1].crossover(new_population[cross_gene], GEN_RATE)
        new_population[-1].fitness(nodes, cost_matrix, depot, CAPACITY, VEHICLE_NUMBER)
    for i in range(CHROMOSOMES_TO_MUTATE):
        gene = randrange(len(new_population))
        new_population[gene].mutate(MUT_RATE, VEHICLE_NUMBER)
        new_population[gene].fitness(nodes, cost_matrix, depot, CAPACITY, VEHICLE_NUMBER)
    manage_duplicates(new_population)
    new_population.sort(key=lambda x: x.score)
    previous_best_solution = new_population[0].score
    population = new_population
    print(f'Generation: {generation}', population[0].score)


for sol in population:
    best = sol
    if best.feasible:
        break

routes = best.create_route_list(nodes, depot, CAPACITY)
sol = TwoOptSolution(routes)
top = TwoOptMove()
tabu = Tabu(10, 20)
best_solution = clone_solution(sol)

for iterator in range(900):
    top.initialize()
    find_best_two_opt_move(top, sol, best_solution, tabu, iterator)
    apply_two_opt_move(top, sol, tabu, iterator)
    if sol.total_cum_cost < best_solution.total_cum_cost:
        best_solution = clone_solution(sol)
    print(f'Iteration: {iterator}, Solution cost: {sol.total_cum_cost}, best Solution cost: {best_solution.total_cum_cost}')


with open('../my_solution.txt', 'w') as f:
    f.write(f'Cost:\n{best_solution.total_cum_cost}\n')
    f.write(f'Routes:\n{VEHICLE_NUMBER}')
    for route in best_solution.routes:
        f.write('\n')
        for node in route.nodes:
            f.write(f'{node.id}')
            if node != route.last():
                f.write(',')


end_time = time.time()
print(f'time elapsed: {end_time - start_time}')
