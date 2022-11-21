import time
from math import sqrt


class Node:
    def __init__(self, id_n=0, x_cor=0, y_cor=0, dem=0, un_time=0):
        self.id = id_n
        self.x = x_cor
        self.y = y_cor
        self.demand = dem
        self.unloading_time = un_time

    def from_line(self, line, sep=','):
        split_line = line.split(sep)
        self.id = int(split_line[0])
        self.x = int(split_line[1])
        self.y = int(split_line[2])
        self.demand = int(split_line[3])
        self.unloading_time = int(split_line[4])

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

    def add_node(self, new_node, minimum_cost):
        self.nodes.append(new_node)
        self.total += minimum_cost
        self.cum += self.total
        self.total += new_node.unloading_time
        self.load += new_node.demand

    def last(self):
        return self.nodes[-1]


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


start_time = time.time()

with open('Instance.txt') as f:
    VEHICLE_NUMBER = get_int_from_line(f.readline())
    CAPACITY = get_int_from_line(f.readline())
    CUSTOMERS = get_int_from_line(f.readline())
    f.readline()  # get rid of NODE INFO
    f.readline()  # get rid of column names
    depot = Node()
    depot.from_line(f.readline())
    nodes = [depot]
    for line in f.readlines():
        node = Node()
        node.from_line(line)
        nodes.append(node)

print(VEHICLE_NUMBER)
print(CAPACITY)
print(CUSTOMERS)

cost_matrix = calculate_cost_matrix(nodes)

routes = [Route(depot, CAPACITY) for _ in range(VEHICLE_NUMBER)]
unserved = nodes[1:]
while unserved:
    route_pos = -1
    node_pos = -1
    min_cum = 10 ** 10
    min_cost = 0
    for pos_n, node in enumerate(unserved):
        for pos_r, route in enumerate(routes):
            from_node = route.last()
            cost = cost_matrix[from_node.id][node.id]
            total = route.total + cost
            if total < min_cum and route.load + node.demand <= route.MAX_LOAD:
                min_cum = total
                min_cost = cost
                route_pos = pos_r
                node_pos = pos_n
    node_to_be_added = unserved.pop(node_pos)
    routes[route_pos].add_node(node_to_be_added, min_cost)

cost = 0
for route in routes:
    cost += route.cum
    string = 'Route: '
    for node in route.nodes:
        string += str(node.id) + ' - '
    string = string.removesuffix(' - ')
    string += f', total={route.total}, cum={route.cum}, load={route.load}'
    print(string)

with open('my_solution.txt', 'w') as f:
    f.write(f'Cost:\n{cost}\n')
    f.write(f'Routes:\n{VEHICLE_NUMBER}')
    for route in routes:
        f.write('\n')
        for node in route.nodes:
            f.write(f'{node.id}')
            if node != route.last():
                f.write(',')


end_time = time.time()
print(f'time elapsed: {end_time - start_time}')
