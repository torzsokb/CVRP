import numpy as np

def read_instance_cvrp(path: str) -> tuple[np.uint16, np.uint16, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "r") as file:
        data = file.read().splitlines()

        n_nodes = np.uint16(data[0].split(" ")[2])
        capacity = np.uint16(data[1].split(" ")[2])

        x = np.zeros(n_nodes, dtype=np.uint16)
        y = np.zeros(n_nodes, dtype=np.uint16)
        demand = np.zeros(n_nodes, dtype=np.uint16)

        shift_coord = 3
        shift_demand = 3 + n_nodes + 1

        for i in range(n_nodes):
            x[i] = np.uint16(data[i + shift_coord].split(" ")[2])
            y[i] = np.uint16(data[i + shift_coord].split(" ")[3])
            demand[i] = np.uint16(data[i + shift_demand].split(" ")[1])

        return n_nodes, capacity, x, y, demand
    

def save_solution(instance_number: int, solution: list[np.ndarray]) -> None:
    
    out = ""

    for route in solution:
        for i in range(len(route)):
            out += f"{route[i]} "
        out += "\n"

    with open(f"outputs/solution_{instance_number}.txt", "w") as f:
        f.write(out)