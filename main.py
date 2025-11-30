import numpy as np
from data_utils import read_instance_cvrp
import plotly.graph_objects as go
from greedy_heur import solve_cvrp_instance_insertion_heur


INSTANCES = [1, 2, 3, 4, 5]

def plot_instance_data(instance_number: int, n_nodes: int, capacity: int, x: np.ndarray, y: np.ndarray, demand: np.ndarray):

        print(f"instance: {instance_number}\tn nodes: {n_nodes}\tcapacity: {capacity}")
        for n in range(n_nodes):
            print(f"node {n+1} x: {x[n]} y: {y[n]}, demand: {demand[n]}")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x[1:], 
            y=y[1:], 
            mode="markers", 
            marker=dict(
                color=demand[1:],
                colorbar=dict(title="Demand"),
                size=7),
            name="Nodes",
            showlegend=False))
        
        fig.add_trace(go.Scatter(
            x=x[:1],
            y=y[:1],
            mode="markers",
            marker=dict(
                color="#66aa00",
                size=13),
            name="Depot"))
        
        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.05,
                x=1.05,
                xanchor="center"),
            title=f"Instance {instance_number} ({n_nodes} nodes, capacity: {capacity})",
            xaxis_title="x coordinate",
            yaxis_title="y coordinate")
        
        fig.show()


def run_instance(instance_number: int, plot_instance: bool=True):
    
    path = f"instances/instance{instance_number}.txt"
    n_nodes, capacity, x, y, demand = read_instance_cvrp(path)

    if plot_instance:
         plot_instance_data(instance_number, n_nodes, capacity, x, y, demand)
    
    solve_cvrp_instance_insertion_heur(capacitiy=capacity, x=x, y=y, demand=demand)


def main():
    for instance in INSTANCES:
        run_instance(instance)

if __name__ == "__main__":
    main()