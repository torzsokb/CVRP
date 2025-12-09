import numpy as np
from numba import njit, prange

@njit
def get_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_nodes = len(x)
    distances = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            distances[i,j] = dist
    return distances

@njit
def min_demand(unvisited: list, demand: np.ndarray) -> np.uint16:
    min_demand = demand.max()
    for node in unvisited:
        if demand[node] < min_demand:
            min_demand = demand[node]
    return np.uint16(min_demand)

@njit
def insert_numba(arr: np.ndarray, idx: np.uint16, value: np.uint16):
    n = arr.shape[0]
    out = np.empty(n + 1, arr.dtype)

    for i in range(idx):
        out[i] = arr[i]

    out[idx] = value

    for i in range(idx, n):
        out[i+1] = arr[i]

    return out

@njit
def choose_random(from_list: list[np.uint16], random_val: np.float64) -> np.uint16:
    interval = np.float64(1.0 / len(from_list))

    if len(from_list) == 1:
        return from_list[0]

    for i, node in enumerate(from_list):
        if (i + 1) * interval >= random_val:
            return node

    return node
    
@njit
def get_objective(distances: np.ndarray, solution: list[np.ndarray]):
    obj = 0.0
    for route in solution:
        for i in range(1, len(route)):
            obj += distances[route[i-1], route[i]]
    return obj

@njit
def route_length(distances: np.ndarray, route: np.ndarray) -> np.float64:
    length = np.float64(0)
    for i in range(1, len(route)):
        length += distances[route[i-1], route[i]]
    return length

@njit
def min_insertion_cost_arr(distances: np.ndarray, route: np.ndarray, candidate_node: np.uint16):
    
    best_index = -1
    min_insertion_cost = distances.max() * 2

    for i in range(1, len(route)):
        curr_node = route[i]
        prev_node = route[i-1]

        cost_arc_to = distances[prev_node, candidate_node]
        cost_arc_from = distances[candidate_node, curr_node]
        cost_arc_skipped = distances[prev_node, curr_node]

        insertion_cost = cost_arc_to + cost_arc_from - cost_arc_skipped

        if insertion_cost < min_insertion_cost:
            min_insertion_cost = insertion_cost
            best_index = i

    return best_index, min_insertion_cost

@njit
def min_insertion_cost_list(distances: np.ndarray, route: list[int], candidate_node: np.uint16):
    
    best_index = 0
    min_insertion_cost = distances.max() * 2

    for i in range(1, len(route)):
        curr_node = route[i]
        prev_node = route[i-1]

        cost_arc_to = distances[prev_node, candidate_node]
        cost_arc_from = distances[candidate_node, curr_node]
        cost_arc_skipped = distances[prev_node, curr_node]

        insertion_cost = cost_arc_to + cost_arc_from - cost_arc_skipped

        if insertion_cost < min_insertion_cost:
            min_insertion_cost = insertion_cost
            best_index = i

    return best_index, min_insertion_cost

@njit
def node_removal_saving(distances: np.ndarray, route: np.ndarray, position: np.uint16) -> np.float64:
    return distances[route[position-1], route[position]] + distances[route[position], route[position+1]] - distances[route[position-1], route[position+1]]

@njit
def relocate_customer_within_route(distances: np.ndarray, solution: list[np.ndarray], demand: np.ndarray, lengths: list[np.float64], loads: list[np.uint16], capacity: np.uint16) -> bool:
    
    for r, route in enumerate(solution):
        for position in range(1, len(route) - 1):
            
            new_pos, saving = max_reinsertion_saving(distances=distances, route=route, current_pos=position)
            if new_pos > 0 and saving > 0 and new_pos != position:
                
                new_route = reinsert_customer(route=route, old_pos=position, new_pos=new_pos)
                
                # this looks inefficient but small numerical errors add over many iterations
                lengths[r] = route_length(distances=distances, route=new_route)
                solution[r] = new_route

                return True
                
    return False
        
@njit 
def reinsert_customer(route: np.ndarray, old_pos: np.uint16, new_pos: np.uint16) -> np.ndarray:

    node = route[old_pos]
    new_route = np.zeros(len(route), dtype=np.uint16)

    for i in range(1, len(route) - 1):
        
        if new_pos < old_pos:
            if i == new_pos:
                new_route[i] = node
            elif i > new_pos and i <= old_pos:
                new_route[i] = route[i-1]
            else:
                new_route[i] = route[i]
                
        if new_pos > old_pos:
            if i == new_pos:
                new_route[i] = node
            elif i >= old_pos and i < new_pos:
                new_route[i] = route[i+1]
            else:
                new_route[i]= route[i]

    return new_route

@njit
def max_reinsertion_saving(distances: np.ndarray, route: np.ndarray, current_pos: np.uint16):
    saving = node_removal_saving(distances=distances, route=route, position=current_pos)
    r_list = []
    
    for i in range(len(route)):
        if not i == current_pos:
            r_list.append(route[i])

    index, cost = min_insertion_cost_list(distances=distances, route=r_list, candidate_node=route[current_pos])
    if saving - cost > 0:
        return index, cost
    return -1, 0.0

# @njit
# def max_reinsertion_saving(distances: np.ndarray, route: np.ndarray, current_pos: np.uint16):

#     node = route[current_pos]
#     best_pos = -1
#     max_saving = 0.0
#     saving = node_removal_saving(distances=distances, route=route, position=current_pos)


#     for pos in range(1, len(route)-1):
#         if pos == current_pos or pos == current_pos + 1:
#             continue

#         reinsertion_cost = distances[route[pos-1], node] + distances[node, route[pos]] - distances[route[pos-1], route[pos]]
#         total_saving = saving - reinsertion_cost
#         if total_saving > max_saving:
#             # best_pos = pos
#             # max_saving = total_saving
#             return pos, total_saving

#     return best_pos, max_saving

@njit
def take_customer_from_other_route_first_impr(distances: np.ndarray, solution: list[np.ndarray], demand: np.ndarray, lengths: list[np.float64], loads: list[np.uint16], capacity: np.uint16) -> bool:
    
    for i, route in enumerate(solution):

        route_nodes = np.zeros(len(route) - 2, dtype=np.uint16)
        route_demands = np.zeros(len(route) - 2, dtype=np.uint16)
        route_savings = np.zeros(len(route) - 2, dtype=np.float64)

        for n in range(1, len(route) - 1):
            route_nodes[n - 1] = route[n]
            route_demands[n - 1] = demand[route[n]]
            route_savings[n - 1] = node_removal_saving(distances, route, n)

        order = np.argsort(route_savings)[::-1]
        route_savings = route_savings[order]
        route_demands = route_demands[order]
        route_nodes = route_nodes[order]

        min_d = route_demands.min()

        for j, route_other in enumerate(solution):
            remaining_capacity = capacity - loads[j]

            if i == j or min_d > remaining_capacity:
                continue

            for k in range(len(route_nodes)):
                if route_demands[k] <= remaining_capacity:
                    index, cost = min_insertion_cost_arr(distances, route_other, candidate_node=route_nodes[k])
                    if route_savings[k] > cost:
                        route_updated = []
                        for n in range(len(route)):
                            if not route[n] == route_nodes[k]:
                                route_updated.append(route[n])
                        solution[i] = np.asarray(route_updated, dtype=np.uint16)
                        solution[j] = insert_numba(route_other, index, route_nodes[k])
                        
                        loads[i] -= route_demands[k]
                        loads[j] += route_demands[k]
                        lengths[i] -= route_savings[k]
                        lengths[j] += cost

                        return True
                    
    return False

@njit
def reverse_part_of_route(distances: np.ndarray, solution: list[np.ndarray], demand: np.ndarray, lengths: list[np.float64], loads: list[np.uint16], capacity: np.uint16) -> bool:

    for r, route in enumerate(solution):
        i, j, saving = part_reversal_saving(distances, route)
        if saving > 0:
            new_route = route.copy()
            while i < j:
                new_route[i], new_route[j] = new_route[j], new_route[i]
                i += 1
                j -= 1
            lengths[r] = route_length(distances, new_route)
            solution[r] = new_route
            return True

    return False

@njit
def part_reversal_saving(distances: np.ndarray, route: np.ndarray):
    
    best_i, best_j = -1, -1
    max_saving = 0.0
    n = len(route)

    for i in range(1, n-2):            
        for j in range(i+1, n-1):   

            old_cost = distances[route[i-1], route[i]] + distances[route[j], route[j+1]]

            new_cost = distances[route[i-1], route[j]] + distances[route[i], route[j+1]]

            saving = old_cost - new_cost
            if saving > max_saving:
                max_saving = saving
                best_i, best_j = i, j

    return best_i, best_j, max_saving            

@njit
def swap_customers_between_routes(distances: np.ndarray, solution: list[np.ndarray], demand: np.ndarray, lengths: list[np.float64], loads: list[np.uint16], capacity: np.uint16) -> bool:
    
    savings = np.zeros(len(demand), dtype=np.float64)
    leftover_caps = np.zeros(len(demand), dtype=np.uint16)
    candidate_route_indices = np.zeros(len(demand), dtype=np.uint16)
    candidate_routes = []

    for r, route in enumerate(solution):
        for pos in range(1, len(route) - 1):

            node = route[pos]
            savings[node] = node_removal_saving(distances, route, pos)
            leftover_caps[node] = capacity - loads[r] + demand[node]
            candidate_route = []

            for i in range(len(route)):
                if not i == pos:
                    candidate_route.append(route[i])

            candidate_route_indices[node] = len(candidate_routes)
            candidate_routes.append(np.asarray(candidate_route, dtype=np.uint16))


    for r1, route1 in enumerate(solution):
        for r2, route2 in enumerate(solution):
            
            if r2 <= r1:
                continue

            for pos1 in range(1, len(route1) - 1):
                node1 = route1[pos1]
                node2_max_demand = leftover_caps[node1]
                saving_route1 = savings[node1]

                for pos2 in range(1, len(route2) - 1):
                    node2 = route2[pos2]
                    node1_max_demand = leftover_caps[node2]
                    saving_route2 = savings[node2]

                    if node1_max_demand < demand[node1] or node2_max_demand < demand[node2]:
                        continue

                    new_route1 = candidate_routes[candidate_route_indices[node1]]
                    new_route2 = candidate_routes[candidate_route_indices[node2]]

                    idx_route1, cost_route1 = min_insertion_cost_arr(distances, new_route1, node2)
                    idx_route2, cost_route2 = min_insertion_cost_arr(distances, new_route2, node1)

                    if saving_route1 + saving_route2 > cost_route1 + cost_route2:

                        new_route1 = insert_numba(new_route1, idx_route1, node2)
                        new_route2 = insert_numba(new_route2, idx_route2, node1)

                        loads[r1] -= demand[node1]
                        loads[r1] += demand[node2]
                        loads[r2] -= demand[node2]
                        loads[r2] += demand[node1]

                        lengths[r1] = route_length(distances, new_route1)
                        lengths[r2] = route_length(distances, new_route2)

                        solution[r1] = new_route1
                        solution[r2] = new_route2

                        return True
    return False

# def swap_customers_between_routes(distances: np.ndarray, solution: list[np.ndarray], demand: np.ndarray, lengths: list[np.float64], loads: list[np.uint16], capacity: np.uint16) -> bool:
    
#     candidate_routes = {}

#     for r, route in enumerate(solution):
        
#         route_dict = {
#             "nodes": [route[i] for i in range(1, len(route) - 1)],
#             "remaining_cap": capacity - loads[r],
#             "without_node" : {}
#             }
        
#         for pos in range(1, len(route) - 1):

#             without_node = {
#                 "remaining_cap": capacity - loads[r] + demand[route[pos]],
#                 "saving": node_removal_saving(distances, route, pos),
#                 "new_route": [route[i] for i in range(len(route)) if not i == pos]
#             }

#             route_dict["without_node"][route[pos]] = without_node

#         candidate_routes[r] = route_dict


#     for r1, route1 in enumerate(solution):
    
#         for r2, route2 in enumerate(solution):
            
#             if r2 <= r1:
#                 continue

#             for node1 in candidate_routes[r1]["nodes"]:

#                 node2_max_demand = candidate_routes[r1]["without_node"][node1]["remaining_cap"]
#                 saving_route1 = candidate_routes[r1]["without_node"][node1]["saving"]

#                 for node2 in candidate_routes[r2]["nodes"]:

#                     node1_max_demand = candidate_routes[r2]["without_node"][node2]["remaining_cap"]
#                     saving_route2 = candidate_routes[r2]["without_node"][node2]["saving"]
                    
#                     if node2_max_demand < demand[node2] or node1_max_demand < demand[node1]:
#                         continue

#                     new_route1 = candidate_routes[r1]["without_node"][node1]["new_route"]
#                     new_route2 = candidate_routes[r2]["without_node"][node2]["new_route"]

#                     idx_route1, cost_route1 = min_insertion_cost_list(distances, new_route1, node2)
#                     idx_route2, cost_route2 = min_insertion_cost_list(distances, new_route2, node1)

#                     if cost_route1 + cost_route2 < saving_route1 + saving_route2:

#                         new_route1.insert(idx_route1, node2)
#                         new_route2.insert(idx_route2, node1)

#                         arr1 = np.asarray(new_route1, dtype=np.uint16)
#                         arr2 = np.asarray(new_route2, dtype=np.uint16)

#                         # This is the dumbest piece of code i have ever written but for some reason this was the only way to ensure correct data types for numba
#                         for r in range(len(solution)):
#                             if r == r1:
#                                 loads[r] = np.uint16(sum(demand[i] for i in new_route1))
#                             elif r == r2:
#                                 loads[r] = np.uint16(sum(demand[i] for i in new_route2))
#                             else:
#                                 loads[r] = np.uint16(capacity - candidate_routes[r]["remaining_cap"])

#                         lengths[r1] = route_length(distances, arr1)
#                         lengths[r2] = route_length(distances, arr2)

#                         solution[r1] = arr1
#                         solution[r2] = arr2

#                         return True

#     return False

@njit
def solve_cvrp_instance_insertion_heur(capacitiy: np.uint16, x: np.ndarray, y: np.ndarray, demand: np.ndarray):
    
    n_nodes = len(x)
    depot = 0
    unvisited = []
    tours = []
    lengths = []
    loads = []

    for i in range(1, n_nodes):
        unvisited.append(int(i))

    distances = get_distances(x=x, y=y)
    max_dist = distances.max() * 2

    while len(unvisited) > 0:

        current_tour = [depot, depot]
        tour_length = np.float64(0)
        remaining_capacity = capacitiy
        min_d = min_demand(unvisited, demand)

        while min_d <= remaining_capacity:
            
            min_insertion_cost = max_dist
            insertion_index = np.uint16(0)
            best_node = np.uint16(0)

            for candidate_node in unvisited:
                if demand[candidate_node] > remaining_capacity:
                    continue

                cancidate_index, cost = min_insertion_cost_list(distances=distances, route=current_tour, candidate_node=candidate_node)
                
                if cost < min_insertion_cost:
                    best_node = candidate_node
                    insertion_index = cancidate_index
                    min_insertion_cost = cost
            
            if not best_node == depot:
                current_tour.insert(insertion_index, best_node)
                unvisited.remove(best_node)
                remaining_capacity -= demand[best_node]
                tour_length += min_insertion_cost
                min_d = min_demand(unvisited, demand)
            else:
                break
                
        tours.append(np.asarray(current_tour, dtype=np.uint16))
        lengths.append(tour_length)
        loads.append(np.uint16(capacitiy - remaining_capacity))

        # print(current_tour)
        # print(tour_length)

    return tours, lengths, loads, distances

@njit
def cvrp_greedy_randomized_insertion_heur(capacitiy: np.uint16, x: np.ndarray, y: np.ndarray, demand: np.ndarray, alpha: np.float64, rand_vals: np.ndarray):
    
    n_nodes = len(x)
    depot = 0
    unvisited = []
    tours = []
    lengths = []
    loads = []

    for i in range(1, n_nodes):
        unvisited.append(int(i))

    distances = get_distances(x=x, y=y)
    max_dist = distances.max() * 2
    
    while len(unvisited) > 0:

        current_tour = [depot, depot]
        tour_length = np.float64(0)
        remaining_capacity = capacitiy
        min_d = min_demand(unvisited, demand)

        while min_d <= remaining_capacity:
            
            insertion_costs = np.full(len(unvisited), fill_value=max_dist, dtype=np.float64)
            insertion_indices = np.zeros(len(unvisited), dtype=np.uint16)
            candidate_nodes = np.zeros(len(unvisited), dtype=np.uint16)

            if len(unvisited) == 1:
                node = unvisited[0]
                if demand[node] > remaining_capacity:
                    break
                else:
                    index, cost = min_insertion_cost_list(distances=distances, route=current_tour, candidate_node=node)
                    current_tour.insert(index, node)
                    unvisited.remove(node)
                    remaining_capacity -= demand[node]
                    tour_length += cost
                    break
            
            for i, candidate_node in enumerate(unvisited):
                if demand[candidate_node] > remaining_capacity:
                    continue

                insertion_index, insertion_cost = min_insertion_cost_list(distances=distances, route=current_tour, candidate_node=candidate_node)
                insertion_costs[i] = insertion_cost
                insertion_indices[i] = insertion_index
                candidate_nodes[i] = candidate_node

            min_insertion_cost = insertion_costs.min()
            best_nodes = []

            for i in range(len(unvisited)):
                if insertion_costs[i] <= (1 + alpha) * min_insertion_cost:
                    best_nodes.append(i)

            selected_node_index = choose_random(best_nodes, rand_vals[len(unvisited) - 1])
            selected_node = candidate_nodes[selected_node_index]

            if not selected_node == depot:
                current_tour.insert(insertion_indices[selected_node_index], selected_node)
                unvisited.remove(selected_node)
                remaining_capacity -= demand[selected_node]
                tour_length += insertion_costs[selected_node_index]
                min_d = min_demand(unvisited, demand)
                if len(unvisited) == 0:
                    break
            else:
                break
        
            
        tours.append(np.asarray(current_tour, dtype=np.uint16))
        lengths.append(tour_length)
        loads.append(np.uint16(capacitiy - remaining_capacity))

        # print(current_tour)
        # print(tour_length)

    return tours, lengths, loads, distances

@njit(nogil=True)
def cvrp_vnd(
    distances: np.ndarray, 
    solution: list[np.ndarray], 
    demand: np.ndarray, 
    lengths: list[np.float64], 
    loads: list[np.uint16], 
    capacity: np.uint16, 
    max_iter: np.uint16, 
    neighbourhood_1_improvement, 
    neighbourhood_2_improvement, 
    neighbourhood_3_improvement, 
    neighbourhood_4_improvement,
    nh_order: list[np.uint16]):
    
    for i in range(max_iter):

        improved = False

        for nh in nh_order:
            
            if nh == 1:
                improved = neighbourhood_1_improvement(distances=distances, solution=solution, demand=demand, lengths=lengths, loads=loads, capacity=capacity)
            elif nh == 2:
                improved = neighbourhood_2_improvement(distances=distances, solution=solution, demand=demand, lengths=lengths, loads=loads, capacity=capacity)
            elif nh == 3:
                improved = neighbourhood_3_improvement(distances=distances, solution=solution, demand=demand, lengths=lengths, loads=loads, capacity=capacity)
            elif nh == 4:
                improved = neighbourhood_4_improvement(distances=distances, solution=solution, demand=demand, lengths=lengths, loads=loads, capacity=capacity)

            if improved:
                break

        if not improved:
            break
     
    # print(get_objective(distances=distances, solution=solution))
    return solution, lengths, loads, i

@njit(nogil=True)
def solve_cvrp_vnd(capacitiy: np.uint16, x: np.ndarray, y: np.ndarray, demand: np.ndarray, max_iter: np.uint16, nh_order: list[np.uint16]):

    solution, lengths, loads, distances = solve_cvrp_instance_insertion_heur(capacitiy=capacitiy, x=x, y=y, demand=demand)
    return cvrp_vnd(
        distances=distances, 
        solution=solution, 
        lengths=lengths, 
        loads=loads, 
        capacity=capacitiy, 
        demand=demand, 
        max_iter=max_iter,
        neighbourhood_1_improvement=take_customer_from_other_route_first_impr,
        neighbourhood_2_improvement=reverse_part_of_route,
        neighbourhood_3_improvement=relocate_customer_within_route,
        neighbourhood_4_improvement=swap_customers_between_routes,
        nh_order=nh_order)

@njit(parallel=False, nogil=True)
def solve_cvrp_grasp(capacitiy: np.uint16, x: np.ndarray, y: np.ndarray, demand: np.ndarray, max_iter_vnd: np.uint16, num_initial_solutions: np.uint16, alpha: np.float64, nh_order: list[np.uint16]):
    
    np.random.seed(42)
    rand_vals = np.random.uniform(low=0.0, high=1.0, size=(num_initial_solutions, len(demand)))
    
    return run_grasp_iterations(
        capacitiy=capacitiy,
        x=x,
        y=y,
        demand=demand,
        max_iter_vnd=max_iter_vnd,
        num_initial_solutions=num_initial_solutions,
        alpha=alpha,
        nh_order=nh_order,
        rand_vals=rand_vals
    )


    
@njit(parallel=True, nogil=True)
def run_grasp_iterations(capacitiy: np.uint16, x: np.ndarray, y: np.ndarray, demand: np.ndarray, max_iter_vnd: np.uint16, num_initial_solutions: np.uint16, alpha: np.float64, nh_order: list[np.uint16], rand_vals: np.ndarray):
    
    objs = np.zeros(num_initial_solutions, dtype=np.float64)
    
    for i in prange(num_initial_solutions):
        solution, lengths, loads, distances = cvrp_greedy_randomized_insertion_heur(capacitiy=capacitiy, x=x, y=y, demand=demand, alpha=alpha, rand_vals=rand_vals[i])
        solution, lengths, loads, iter = cvrp_vnd(
            distances=distances, 
            solution=solution, 
            lengths=lengths, 
            loads=loads, 
            capacity=capacitiy, 
            demand=demand, 
            max_iter=max_iter_vnd,
            neighbourhood_1_improvement=take_customer_from_other_route_first_impr,
            neighbourhood_2_improvement=reverse_part_of_route,
            neighbourhood_3_improvement=relocate_customer_within_route,
            neighbourhood_4_improvement=swap_customers_between_routes,
            nh_order=nh_order)
        objs[i] = get_objective(distances, solution)

    best_trail = np.argmin(objs)
    solution, lengths, loads, distances = cvrp_greedy_randomized_insertion_heur(capacitiy=capacitiy, x=x, y=y, demand=demand, alpha=alpha, rand_vals=rand_vals[best_trail])
    return cvrp_vnd(
        distances=distances, 
        solution=solution, 
        lengths=lengths, 
        loads=loads, 
        capacity=capacitiy, 
        demand=demand, 
        max_iter=max_iter_vnd,
        neighbourhood_1_improvement=take_customer_from_other_route_first_impr,
        neighbourhood_2_improvement=reverse_part_of_route,
        neighbourhood_3_improvement=relocate_customer_within_route,
        neighbourhood_4_improvement=swap_customers_between_routes,
        nh_order=nh_order)