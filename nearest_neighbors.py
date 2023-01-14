def construct_solution(depot, capacity, vehicles, customers, nodes, matrix):
    avg_customers_with_depot = customers / vehicles + 1
    routes = [Route(depot, capacity, avg_customers_with_depot) for _ in range(vehicles)]
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
