import numpy as np
from data_utils import read_instance_cvrp
import plotly.graph_objects as go
from greedy_heur import solve_cvrp_instance_insertion_heur
from vnd import solve_cvrp_vnd


INSTANCES = [1, 2, 3, 4, 5]

def plot_instance_data(instance_number: int, 
                       n_nodes: int, 
                       capacity: int, 
                       x: np.ndarray, 
                       y: np.ndarray, 
                       demand: np.ndarray,
                       routes: list[list[int]]=None,
                       lengths: list[float]=None,
                       loads: list[int]=None):

        print(f"instance: {instance_number}\tn nodes: {n_nodes}\tcapacity: {capacity}")
        # for n in range(n_nodes):
        #     print(f"node {n+1} x: {x[n]} y: {y[n]}, demand: {demand[n]}")

        title = f"Instance {instance_number} ({n_nodes} nodes, capacity: {capacity}"
        if lengths is not None:
            title += f" objective: {sum(lengths):.2f})"
        else:
            title += ")"

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
                 size=7,
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
    
    path = f"instances/instance{instance_number}.txt"
    n_nodes, capacity, x, y, demand = read_instance_cvrp(path)

    routes, lengths, loads = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=0)
    if plot_instance:
         plot_instance_data(instance_number, n_nodes, capacity, x, y, demand, routes, lengths, loads)
    
    # routes, lengths, loads = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=1)
    # if plot_instance:
    #      plot_instance_data(instance_number, n_nodes, capacity, x, y, demand, routes, lengths, loads)
    
    # routes, lengths, loads = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=5)
    # if plot_instance:
    #      plot_instance_data(instance_number, n_nodes, capacity, x, y, demand, routes, lengths, loads)

    routes, lengths, loads = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=10)
    if plot_instance:
         plot_instance_data(instance_number, n_nodes, capacity, x, y, demand, routes, lengths, loads)
    
    routes, lengths, loads = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=100)
    if plot_instance:
         plot_instance_data(instance_number, n_nodes, capacity, x, y, demand, routes, lengths, loads)

    # routes, lengths, loads = solve_cvrp_vnd(capacitiy=capacity, x=x, y=y, demand=demand, max_iter=1000)
    # if plot_instance:
    #      plot_instance_data(instance_number, n_nodes, capacity, x, y, demand, routes, lengths, loads)
    
    
    
    
    
    


def main():
    for instance in INSTANCES:    
        run_instance(instance)

if __name__ == "__main__":
    main()