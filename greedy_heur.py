import numpy as np
from numba import njit


@njit
def solve_cvrp_instance_insertion_heur(capacitiy: np.uint16, x: np.ndarray, y: np.ndarray, demand: np.ndarray):
    
    n_nodes = len(x)
    depot = 0
    unvisited = []
    tours = []

    for i in range(1, n_nodes):
        unvisited.append(int(i))

    distances = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            distances[i,j] = dist
    max_dist = distances.max() * 2

    while len(unvisited) > 0:

        current_tour = [depot]
        tour_length = np.float64(0)
        remaining_capacity = capacitiy
        min_dist = max_dist
        closest = 0

        for node in unvisited:
            if not node == depot:
                if distances[node, depot] < min_dist:
                    min_dist = distances[node, depot]
                    closest = node

        current_tour.append(closest)
        current_tour.append(depot)
        unvisited.remove(closest)
        tour_length += min_dist
        remaining_capacity -= demand[closest]

        min_demand = get_min_demand(unvisited, demand)

        while min_demand <= remaining_capacity:
            
            min_insertion_cost = max_dist * 2
            insertion_index = 0
            best_node = 0

            for candidate_node in unvisited:
                if demand[candidate_node] > remaining_capacity:
                    continue
                
                for i, node in enumerate(current_tour):
                    next_node = current_tour[i+1]
                    
                    cost_arc_to = distances[candidate_node, node]
                    cost_arc_from = distances[candidate_node, next_node]
                    cost_arc_skipped = distances[node, next_node]

                    insertion_cost = cost_arc_to + cost_arc_from - cost_arc_skipped

                    if insertion_cost < min_insertion_cost:
                        min_insertion_cost = insertion_cost
                        insertion_index = i + 1
                        best_node = candidate_node

                    if next_node == depot:
                        break
            
            if best_node == depot:
                break
            else:
                current_tour.insert(insertion_index, best_node)
                unvisited.remove(best_node)
                remaining_capacity -= demand[best_node]
                tour_length += min_insertion_cost
                min_demand = get_min_demand(unvisited, demand)

        tours.append(np.asarray(current_tour, dtype=np.uint16))
        print(current_tour)
        print(tour_length)

    return tours


@njit
def get_min_demand(unvisited: list, demand: np.ndarray) -> np.uint16:
    min_demand = demand.max()
    for node in unvisited:
        if demand[node] < min_demand:
            min_demand = demand[node]
    return min_demand






    


    