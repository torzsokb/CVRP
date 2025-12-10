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

def reformat_solutions():
    for i in range(1, 6):
        new_solution = ""

        with open(f"outputs/solution_{i}.txt", "r") as f:
            old_solution = f.readlines()

            for route in old_solution:
                nodes = route.split()
                for node in nodes:
                    new_solution += f"{int(node) + 1} "
                new_solution += "\n"

        with open(f"outputs/solution_{i}.txt", "w") as f:
            f.write(new_solution)

def main():
    reformat_solutions()

if __name__ == "__main__":
    main()