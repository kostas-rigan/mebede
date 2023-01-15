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


def main():
    # Components from main
    population_size = 1000
    generations = 700
    mutation_rate = 0.5
    gene_selection_rate = 0.3
