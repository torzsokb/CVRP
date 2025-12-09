import numpy as np
from data_utils import read_instance_cvrp
import plotly.graph_objects as go
from cvrp_solver import solve_cvrp_vnd, solve_cvrp_grasp
import pandas as pd
import time
from itertools import permutations


# INSTANCES = [1, 2, 3, 4, 5]
INSTANCES = [2]
NH_ORDER = [4, 1, 2, 3]

def plot_instance_data(
          n_nodes: int, 
          x: np.ndarray, 
          y: np.ndarray, 
          demand: np.ndarray,
          title: str,
          routes: list[list[int]]=None,
          lengths: list[float]=None,
          loads: list[int]=None):

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x[1:], 
            y=y[1:], 
            customdata=np.stack((np.arange(1, n_nodes), demand[1:]), axis=1),
            hovertemplate=(
                "Node: %{customdata[0]}<br>"
                "Demand: %{customdata[1]}<br>"
                "x: %{x}<br>"
                "y: %{y}<extra></extra>"
            ),
            mode="markers",
            marker=dict(
                 color=demand[1:],
                 size=10,
                 colorbar=dict(
                      title="Demand",
                      x=-0.1,
                      xanchor="center")),
            name="Customers",
            showlegend=False))
        
        fig.add_trace(go.Scatter(
            x=x[:1],
            y=y[:1],
            mode="markers",
            marker=dict(
                color="#66aa00",
                size=13),
            name="Depot"))
        
        if routes is not None:
            
            for idx, route in enumerate(routes):
                route_x = [x[node] for node in route]
                route_y = [y[node] for node in route]
                custom = np.column_stack([
                    np.full(len(route), idx+1),        # route number
                    np.full(len(route), lengths[idx]), # route length
                    np.full(len(route), loads[idx])    # route load
                ])

                fig.add_trace(go.Scatter(
                    x=route_x,
                    y=route_y,
                    customdata=custom,
                    hovertemplate=(
                         "Route: %{customdata[0]}<br>"
                         "Length: %{customdata[1]:.2f}<br>"
                         "Load: %{customdata[2]}<extra></extra>"
                    ),
                    mode="lines+markers",
                    line=dict(width=2),
                    name=f"Route {idx+1}"
                ))

        
        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.05,
                x=1.05,
                xanchor="center"),
            title=title,
            xaxis_title="x coordinate",
            yaxis_title="y coordinate")
        
        fig.show()

def run_instance(instance_number: int, plot_instance: bool=True):
    
    print(f"instance {instance_number}")
    
    path = f"instances/instance{instance_number}.txt"
    n_nodes, capacity, x, y, demand = read_instance_cvrp(path)
    
    performance = {"name": [], "n_starting_points": [], "alpha": [], "objective": [], "running_time": [], "n_iter": [], "nh_order": []}
    

    # greedy without vnd

    start = time.time()
    routes, lengths, loads, iter = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=0, nh_order=NH_ORDER)
    end = time.time()

    obj = sum(lengths)

    
    if plot_instance:
        title = f"Instance {instance_number} Greedy Solution ({n_nodes} nodes, objectve: {obj:.2f})"
        plot_instance_data(n_nodes, x, y, demand, title, routes, lengths, loads)
    
    performance["name"].append("greedy_insertion")
    performance["n_starting_points"].append(0)
    performance["alpha"].append(0)
    performance["running_time"].append(end - start)
    performance["objective"].append(obj)
    performance["n_iter"].append(0)
    performance["nh_order"].append("0000")


    # greedy with vnd

    nh_orders = list(permutations(NH_ORDER))
    best_obj = 999999
    best_nh_order = [4, 1, 2, 3]

    for nh_order in nh_orders:

        start = time.time()
        routes, lengths, loads, iter = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=1000, nh_order=nh_order)
        end = time.time()

        obj = sum(lengths)

        performance["name"].append("vnd")
        performance["n_starting_points"].append(1)
        performance["alpha"].append(0)
        performance["running_time"].append(end - start)
        performance["objective"].append(obj)
        performance["n_iter"].append(iter)
        performance["nh_order"].append(",".join(map(str, nh_order)))

        if obj < best_obj:
            best_obj = obj
            best_nh_order = nh_order

        print(f"nh order: {nh_order}, obj: {obj:.2f}, time: {(end - start):.5f}")
    
    
    if plot_instance:
        routes, lengths, loads, iter = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=1000, nh_order=best_nh_order)
        title = f"Instance {instance_number} VND Solution ({n_nodes} nodes, objectve: {sum(lengths):.2f}, iterations: {iter}, nh order: {best_nh_order})"
        plot_instance_data(n_nodes, x, y, demand, title, routes, lengths, loads)

    
    # grasp
    print("grasp")    

    solution_sizes = [50, 100, 500, 1000]
    alphas = [0.3, 0.4]
    best_obj = 9999999
    best_alpha = 0.35
    best_size = 900
    # best_nh_order = [2, 4, 1, 3]

    # for solution_size in solution_sizes:
    #     for alpha in alphas:
    #         start = time.time()
    #         routes, lengths, loads, iter = solve_cvrp_grasp(capacitiy=capacity, x=x, y=y, demand=demand, max_iter_vnd=1000, num_initial_solutions=solution_size, alpha=alpha, nh_order=NH_ORDER)
    #         end = time.time()
    #         obj = sum(lengths)

    #         performance["name"].append("grasp")
    #         performance["n_starting_points"].append(solution_size)
    #         performance["alpha"].append(alpha)
    #         performance["running_time"].append(end - start)
    #         performance["objective"].append(obj)
    #         performance["n_iter"].append(iter)

    #         if obj < best_obj:
    #             best_alpha = alpha
    #             best_size = solution_size
    #             best_obj = obj

    for nh_order in nh_orders:
        # if nh_order[0] == 4:
        #     continue
        for alpha in alphas:
            order = nh_order

            start = time.time()
            routes, lengths, loads, iter = solve_cvrp_grasp(capacitiy=capacity, x=x, y=y, demand=demand, max_iter_vnd=1000, num_initial_solutions=best_size, alpha=alpha, nh_order=order)
            end = time.time()

            obj = sum(lengths)

            performance["name"].append("grasp")
            performance["n_starting_points"].append(best_size)
            performance["alpha"].append(alpha)
            performance["running_time"].append(end - start)
            performance["objective"].append(obj)
            performance["n_iter"].append(iter)
            performance["nh_order"].append(",".join(map(str, order)))

            print(f"nh order: {order}, obj: {obj:.2f}, alpha: {alpha}, time: {(end - start):.5f}")

        # if obj < best_obj:
        #     best_obj = obj
        #     best_alpha = alpha
            # best_nh_order = nh_order



    if plot_instance:
        # routes, lengths, loads, iter = solve_cvrp_grasp(capacitiy=capacity, x=x, y=y, demand=demand, max_iter_vnd=1000, num_initial_solutions=best_size, alpha=best_alpha, nh_order=order)
        title = f"Instance {instance_number} GRASP Solution ({n_nodes} nodes, objectve: {obj:.2f}, n: {best_size}, alpha: {best_alpha}, iterations: {iter}, nh order: {NH_ORDER})"
        plot_instance_data(n_nodes, x, y, demand, title, routes, lengths, loads)



    perf = pd.DataFrame.from_dict(performance)
    perf.to_csv(f"outputs/instance_{instance_number}_performance_metrics.csv", index=False)
    # with open(f"outputs/instance_{instance_number}_info.json", "w") as f:
    #     json.dump(info, f)

    


def main():
    for instance in INSTANCES:    
        run_instance(instance, plot_instance=False)

if __name__ == "__main__":
    main()