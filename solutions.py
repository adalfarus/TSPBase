# ------------------------------------------------------
# Has all the solutions to the TSP. Every solution uses
# TPSBase.
# ------------------------------------------------------
import time
from tsp_base import TSPBase
import random
import math


class WanderingSalesmanPersistent(TSPBase):
    def __init__(self, num_points: int = 10):
        super().__init__(num_points, title_text_add_in=" Brute Force")

    def create_initial_task(self):  # Is automatically returned as a tuple
        return self.solve, ([self.points[0]],), {}

    def solve(self, curr_solution: list, curr_length: int = 0):
        if len(curr_solution) == len(self.points):
            # Add the stretch from the last point back to the starting point
            return_length = self.graph[curr_solution[-1]][curr_solution[0]]
            total_length = curr_length + return_length

            self.update_solution(curr_solution, total_length)
            if self.is_final_solution():
                return True
        else:
            for point in [x for x in self.points.copy() if x not in curr_solution]:
                new_length = curr_length + self.graph[curr_solution[-1]][point]
                self.local_task_queue.put((self.solve, (curr_solution + [point], new_length), {}))
                self.local_solutions_checked += 1


if __name__ == "__main__":
    try:
        WanderingSalesmanPersistent(10)
    except KeyboardInterrupt:
        print("Continuing with next solution ...")


class WanderingSalesmanPersistentPlus(TSPBase):
    def __init__(self, num_points: int = 10):
        super().__init__(num_points, title_text_add_in=" Brute Force+")

    def create_initial_task(self):  # Is automatically returned as a tuple
        return self.solve, ([self.points[0]],), {}

    def solve(self, curr_solution: list, curr_length: int = 0):
        if len(curr_solution) == len(self.points):
            # Add the stretch from the last point back to the starting point
            return_length = self.graph[curr_solution[-1]][curr_solution[0]]
            total_length = curr_length + return_length

            self.update_solution(curr_solution, total_length)
            if self.is_final_solution():
                return True
        else:
            for point in [x for x in self.points.copy() if x not in curr_solution]:
                new_length = curr_length + self.graph[curr_solution[-1]][point]
                if (new_length < self.local_best_length and new_length <=  # Within 20% of the better lower bound
                        ((self.better_lower_bound // 10) * 2 + self.better_lower_bound)):
                    self.local_task_queue.put((self.solve, (curr_solution + [point], new_length), {}))
                self.local_solutions_checked += 1


if __name__ == "__main__":
    try:
        WanderingSalesmanPersistentPlus(10)
    except KeyboardInterrupt:
        print("Continuing with next solution ...")


class SuperLazySalesman(TSPBase):
    def __init__(self, num_points: int = 10):
        super().__init__(num_points, title_text_add_in=" NN")

    def create_initial_task(self):  # Is automatically returned as a tuple
        self.local_best_solution = [self.points[0]]
        self.local_best_length = 0
        return self.solve, (), {}

    def solve(self):
        if len(self.local_best_solution) == len(self.points):
            self.local_best_length += self.graph[self.local_best_solution[0]][self.local_best_solution[-1]]
            self.local_solutions_checked += 1
            self.update_solution(self.local_best_solution)
            self.local_best_solution.append(self.local_best_solution[0])
            return True
        else:
            costs = sorted([x for x in self.graph[self.local_best_solution[-1]].items()
                            if x[0] not in self.local_best_solution],
                           key=lambda x: x[1])
            vertex, cost = costs[0]
            self.local_best_solution.append(vertex)
            self.local_best_length += cost
            self.local_task_queue.put((self.solve, (), {}))


if __name__ == "__main__":
    try:
        SuperLazySalesman(10)
    except KeyboardInterrupt:
        print("Continuing with next solution ...")


class SuperLazySalesmanKOpt(TSPBase):
    def __init__(self, num_points: int = 10):
        super().__init__(num_points, title_text_add_in=" NN (+2-opt)")

    def create_initial_task(self):  # Is automatically returned as a tuple
        self.local_best_solution = [self.points[0]]
        self.local_best_length = 0
        return self.solve, (), {}

    def start(self, debug_mode: bool = False):
        super().start(True)

    def solve(self):
        if len(self.local_best_solution) == len(self.points):
            self.local_best_length += self.graph[self.local_best_solution[0]][self.local_best_solution[-1]]
            self.local_solutions_checked += 1
            self.local_best_solution = self.k_opt(self.local_best_solution)
            self.update_solution(self.local_best_solution, update_last=False)
            self.local_best_solution.append(self.local_best_solution[0])
            return True
        else:
            costs = sorted([x for x in self.graph[self.local_best_solution[-1]].items()
                            if x[0] not in self.local_best_solution],
                           key=lambda x: x[1])
            vertex, cost = costs[0]
            self.local_best_solution.append(vertex)
            self.local_best_length += cost
            self.local_task_queue.put((self.solve, (), {}))


if __name__ == "__main__":
    try:
        SuperLazySalesmanKOpt(10)
    except KeyboardInterrupt:
        print("Continuing with next solution ...")


class LazySalesmanSafe(TSPBase):
    def __init__(self, num_points: int = 10):
        super().__init__(num_points, title_text_add_in=" NN+B (Safe)")

    def create_initial_task(self):  # Is automatically returned as a tuple
        self.local_best_solution = [self.points[0]]
        self.local_best_length = 0
        return self.solve, (), {}

    def start(self, debug_mode: bool = False):
        super().start(debug_mode=True)

    def worker(self, _: int = 1000, __: int = 10000):
        super().worker(1, 1)  # For better runtime pass the original values

    def backtrack(self, plus: int = -1, initial_backtrack: int = 0, backtrack_lst: list = None):
        """
        Find the first element from the back that is bt < i-1. So [0, 0, 1, 0, 0] & [3, 2, 1, 0, -1]
                                                                      ^					^
        Calculate length of the Points that need to be backtracked. In this case 2.
        Replace all elements after that with 0. So [0, 0, 1, 0, 0] -> [0, 0, 0, 0, 0]
        Increment the counter of the selected element. So [0, 0, 0, 0, 0] -> [0, 1, 0, 0, 0]
        Use the calculated length of Points that need to be backtracked to remove them from self.local_best_solution.
        Recalculate the length of the remaining points.
        So:
        initial_backtrack = 2
        backtrack_lst = [0, 2, 0, 1, 0] ([3, 2, 1, 0, -1])
                               ^
        self.local_best_solution = [Point(), Point(), Point(), Point()]

        ->

        backtrack = 0
        backtrack_lst = [1, 0, 0, 0, 0] ([3, 2, 1, 0, -1])
                         ^
        self.local_best_solutions = [Point()]
        """
        if backtrack_lst is None:
            backtrack_lst = [0] * len(self.points)

        plus = plus + 1
        initial_backtrack = initial_backtrack + plus
        current_backtrack = None
        for i, bt in enumerate(backtrack_lst[initial_backtrack - 1::-1]):
            if bt < (i - 1 + (len(backtrack_lst) - initial_backtrack)):
                current_backtrack = i
                break

        current_backtrack = (initial_backtrack-1) - current_backtrack

        self.local_best_length = sum([self.graph[point][self.local_best_solution[i+1]]
                                      for i, point in enumerate(self.local_best_solution[:current_backtrack-1])])
        backtrack_lst[current_backtrack] += 1  # Tell it to backtrack one further

        # Reset any values after current_backtrack, but not current_backtrack
        backtrack_lst[current_backtrack+1:] = [0] * len(backtrack_lst[current_backtrack+1:])
        self.local_best_solution = self.local_best_solution[:current_backtrack+1]
        self.local_task_queue.put(
            (self.solve, (current_backtrack, backtrack_lst), {})
        )
        self.local_solutions_checked += 1

    def solve(self, backtrack: int = 0, backtrack_lst=None):
        if backtrack_lst is None:
            backtrack_lst = [0] * len(self.points)

        if len(self.local_best_solution) == len(self.points):
            # Add the stretch from the last point back to the starting point
            return_length = self.graph[self.local_best_solution[-1]][self.local_best_solution[0]]
            total_length = self.local_best_length + return_length

            self.local_best_length = total_length
            self.local_best_solution.append(self.local_best_solution[0])
            if not total_length <= ((self.better_lower_bound // 10) * 2 + self.better_lower_bound):
                self.backtrack(-3, backtrack, backtrack_lst)
                return False
            self.local_solutions_checked += 1
            return True
        else:
            costs = sorted([x for x in self.graph[self.local_best_solution[-1]].items()
                            if x[0] not in self.local_best_solution],
                           key=lambda x: x[1])

            chosen_point, minimum_cost = costs[0+backtrack_lst[backtrack]]
            new_length = self.local_best_length + minimum_cost

            if new_length <= ((self.better_lower_bound // 10) * 2 + self.better_lower_bound):
                self.local_best_length = new_length
                self.local_best_solution.append(chosen_point)
                self.local_task_queue.put((self.solve, (backtrack+1, backtrack_lst), {}))
            else:
                self.backtrack(-1, backtrack, backtrack_lst)
        time.sleep(0.01)


if __name__ == "__main__":
    try:
        LazySalesmanSafe(10)
    except KeyboardInterrupt:
        print("Continuing with next solution ...")


class LazySalesman(TSPBase):
    def __init__(self, num_points: int = 10):
        super().__init__(num_points, title_text_add_in=" NN+B (Faster)")

    def create_initial_task(self):  # Is automatically returned as a tuple
        self.local_best_solution = [self.points[0]]
        self.local_best_length = 0
        return self.solve, (), {}

    def start(self, debug_mode: bool = False):
        super().start(debug_mode=False)

    def worker(self, _: int = 1000, __: int = 10000):
        super().worker(1000, 10000)
        # super().worker(1, 1)  # For better runtime pass the original values

    def backtrack(self, plus: int = -1, initial_backtrack: int = 0, backtrack_lst: list = None):
        """
        Find the first element from the back that is bt < i-1. So [0, 0, 1, 0, 0] & [3, 2, 1, 0, -1]
                                                                      ^					^
        Calculate length of the Points that need to be backtracked. In this case 2.
        Replace all elements after that with 0. So [0, 0, 1, 0, 0] -> [0, 0, 0, 0, 0]
        Increment the counter of the selected element. So [0, 0, 0, 0, 0] -> [0, 1, 0, 0, 0]
        Use the calculated length of Points that need to be backtracked to remove them from self.local_best_solution.
        Recalculate the length of the remaining points.
        So:
        initial_backtrack = 2
        backtrack_lst = [0, 2, 0, 1, 0] ([3, 2, 1, 0, -1])
                               ^
        self.local_best_solution = [Point(), Point(), Point(), Point()]

        ->

        backtrack = 0
        backtrack_lst = [1, 0, 0, 0, 0] ([3, 2, 1, 0, -1])
                         ^
        self.local_best_solutions = [Point()]
        """
        if backtrack_lst is None:
            backtrack_lst = [0] * len(self.points)

        plus = plus + 1
        #print(initial_backtrack, plus, "->", initial_backtrack + plus)
        initial_backtrack = initial_backtrack + plus
        current_backtrack = None

        #print("Backtrack: Starting")
        #print(f"Initial Backtrack: {initial_backtrack}, Plus: {plus}")
        #print(f"Backtrack List (Before): {backtrack_lst}")

        #sol = []
        for i, bt in enumerate(backtrack_lst[initial_backtrack-1::-1]):
            #sol.append((bt, (i - 1 + (len(backtrack_lst) - initial_backtrack))))
            if bt < (i-1):#(i - 1 + (len(backtrack_lst) - initial_backtrack)):
                current_backtrack = i
                break  # backtrack a lot at first in hopes of finding a better solution quick

        if current_backtrack is None:
            for i, bt in enumerate(backtrack_lst[initial_backtrack - 1::-1]):
                # sol.append((bt, (i - 1 + (len(backtrack_lst) - initial_backtrack))))
                #print(backtrack_lst)
                #print([len(backtrack_lst) - i - 2 for i, _ in enumerate(backtrack_lst)])
                #print("   " * (initial_backtrack - 1 - i), "^", f"{(i - 1 + (len(backtrack_lst) - initial_backtrack))}, ({bt})")
                if bt < (i - 1 + (len(backtrack_lst) - initial_backtrack)):  # If a lot of backtrack isn't working
                    current_backtrack = i                                    # Only use normal backtrack
                    break

        #print("NoSol", backtrack_lst, sol, backtrack_lst[initial_backtrack-1::-1], initial_backtrack-1, current_backtrack, plus)
        #if current_backtrack is not None:
        current_backtrack = (initial_backtrack-1) - current_backtrack
        #    print(current_backtrack)
        #else:
        #    return

        # Calculate new length incl the current backtrack
        #print(current_backtrack)
        #print([(point, self.local_best_solution[-i-2]) for i, point in enumerate(self.local_best_solution[:current_backtrack:-1])])
        #print(self.local_best_solution[::-1])
        self.local_best_length = self.local_best_length - sum([self.graph[point][self.local_best_solution[-i - 2]]
                                                               for i, point in enumerate(self.local_best_solution
                                                                                         [:current_backtrack-1:-1])
                                                               if i+1 < len(self.local_best_solution)])
        #print(self.local_best_length)
        #local_best_length = sum([self.graph[point][self.local_best_solution[i+1]] for i, point in enumerate(self.local_best_solution[:current_backtrack-1])]) #self.local_best_length - sum([self.graph[point][self.local_best_solution[-i-2]]
                                 #                              for i, point in enumerate(self.local_best_solution[:current_backtrack+1:-1])])
        #print("1.", self.local_best_length, "\n2.", local_best_length)
        backtrack_lst[current_backtrack] += 1  # Tell it to backtrack one further
        # Reset any values after current_backtrack, but not current_backtrack
        backtrack_lst[current_backtrack+1:] = [0] * len(backtrack_lst[current_backtrack+1:])
        self.local_best_solution = self.local_best_solution[:current_backtrack+1]
        self.local_task_queue.put(
            (self.solve, (current_backtrack, backtrack_lst), {})
        )
        self.local_solutions_checked += 1

        #print(f"Current Backtrack: {current_backtrack}")
        #print(f"New Best Length: {self.local_best_length}")
        #print(f"Backtrack List (After): {backtrack_lst}")
        #print(f"Best Solution (After Backtrack): {len(self.local_best_solution)}")
        #print("Backtrack: Ending\n")

    def solve(self, backtrack: int = 0, backtrack_lst=None):
        if backtrack_lst is None:
            backtrack_lst = [0] * len(self.points)

        if len(self.local_best_solution) == len(self.points):
            # Add the stretch from the last point back to the starting point
            return_length = self.graph[self.local_best_solution[-1]][self.local_best_solution[0]]
            total_length = self.local_best_length + return_length

            self.local_best_length = total_length
            self.local_best_solution.append(self.local_best_solution[0])
            if not total_length <= ((self.better_lower_bound // 10) * 2 + self.better_lower_bound):
                self.backtrack(-3, backtrack, backtrack_lst)
                return False
            self.local_solutions_checked += 1
            return True
        else:
            costs = sorted([x for x in self.graph[self.local_best_solution[-1]].items()
                            if x[0] not in self.local_best_solution],
                           key=lambda x: x[1])

            chosen_point, minimum_cost = costs[0+backtrack_lst[backtrack]]
            new_length = self.local_best_length + minimum_cost

            if new_length <= ((self.better_lower_bound // 10) * 2 + self.better_lower_bound):
                self.local_best_length = new_length
                self.local_best_solution.append(chosen_point)
                self.local_task_queue.put((self.solve, (backtrack+1, backtrack_lst), {}))
            else:
                self.backtrack(-1, backtrack, backtrack_lst)
        #time.sleep(0.001)


if __name__ == "__main__":
    try:
        LazySalesman(10)
    except KeyboardInterrupt:
        print("Continuing with next solution ...")


class Ant:
    def __init__(self, start_index, num_points):
        self.path = [start_index]
        self.total_distance = 0
        self.num_points = num_points

    def select_next_city(self, pheromone_matrix, distance_matrix, alpha, beta):
        current_city = self.path[-1]
        probabilities = []

        for i in range(self.num_points):
            if i not in self.path:
                tau = pheromone_matrix[current_city][i] ** alpha  # Pheromone level
                eta = (1.0 / distance_matrix[current_city][i]) ** beta  # Desirability (inverse of distance)
                probabilities.append(tau * eta)
            else:
                probabilities.append(0)

        total = sum(probabilities)
        if total == 0:
            # Avoid division by zero
            return random.choice([i for i in range(self.num_points) if i not in self.path])

        probabilities = [p / total for p in probabilities]  # Normalize probabilities
        return random.choices(range(self.num_points), weights=probabilities, k=1)[0]

    def add_to_path(self, city, distance):
        self.path.append(city)
        self.total_distance += distance


class WanderingAntColony(TSPBase):
    def __init__(self, num_points: int = 10, num_ants: int = 10, alpha: float = 1.0, beta: float = 1.0, evaporation_rate: float = 0.5, iterations: int = 100):
        super().__init__(num_points, title_text_add_in=" ACO", gui_start=False)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.pheromone_matrix = [[1.0 for _ in range(num_points)] for _ in range(num_points)]

        self.start_gui()

    def create_initial_task(self):
        # Create and return the initial task
        return self.solve_aco, (), {}

    def solve_aco(self):
        best_solution = None
        best_distance = float('inf')

        for iteration in range(self.iterations):
            ants = [Ant(random.randint(0, self.num_points - 1), self.num_points) for _ in range(self.num_ants)]

            # Let each ant build a tour
            for ant in ants:
                while len(ant.path) < self.num_points:
                    next_city = ant.select_next_city(self.pheromone_matrix, self.distance_matrix, self.alpha, self.beta)
                    ant.add_to_path(next_city, self.distance_matrix[ant.path[-2]][next_city])

                # Complete tour by returning to start
                ant.add_to_path(ant.path[0], self.distance_matrix[ant.path[-1]][ant.path[0]])

                # Update best solution
                if ant.total_distance < best_distance:
                    best_solution = ant.path
                    best_distance = ant.total_distance

            # Update pheromones
            self.update_pheromones(ants)

        # Once the iterations are complete, update the best solution found
        self.update_solution(best_solution, best_distance)

        return best_solution, best_distance

    def update_pheromones(self, ants):
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.pheromone_matrix[i][j] *= (1 - self.evaporation_rate)

        for ant in ants:
            contribution = 1.0 / ant.total_distance
            for i in range(len(ant.path) - 1):
                self.pheromone_matrix[ant.path[i]][ant.path[i + 1]] += contribution
                self.pheromone_matrix[ant.path[i + 1]][ant.path[i]] += contribution  # Assuming symmetry


# Example usage
if __name__ == "__main__":
    pass
    #colony = WanderingAntColony(num_points=10, num_ants=10, iterations=100)
    #best_path, best_distance = colony.solve()
    #print(f"Best Path: {best_path}, Distance: {best_distance}")


class NeighborhoodSalesmanNetwork(TSPBase):
    def __init__(self, num_points: int, max_group_size: int = 9, distance_threshold: float = 5.0):
        super().__init__(num_points, title_text_add_in=" Cluster Search", gui_start=False)
        self.max_group_size = max_group_size
        self.distance_threshold = distance_threshold

        self.clusters = []
        self.cluster_solutions = []
        self.start_gui()

    def create_initial_task(self):
        return self.cluster_points, (), {}

    def start(self, _: bool = False):
        super().start(debug_mode=True)

    def cluster_points_into_groups(self):
        used_points = set()

        for point in self.points:
            if point in used_points:
                continue

            cluster = [point]
            used_points.add(point)

            for other_point in self.points:
                if other_point in used_points:
                    continue

                if all(self.graph[cluster_point][other_point] > self.distance_threshold for cluster_point in
                       cluster):
                    continue

                cluster.append(other_point)
                used_points.add(other_point)

                if len(cluster) >= self.max_group_size:
                    break

            self.clusters.append(cluster)

    def find_closest_edges_between_clusters(self):
        closest_edges = []

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                min_distance = float('inf')
                closest_pair = None

                for point_a in self.clusters[i]:
                    for point_b in self.clusters[j]:
                        distance = self.graph[point_a][point_b]
                        if distance < min_distance:
                            min_distance = distance
                            closest_pair = (point_a, point_b)

                closest_edges.append(closest_pair)

        return closest_edges

    def cluster_points(self):
        # Your existing clustering logic
        self.cluster_points_into_groups()

        # Enqueue the first cluster to be solved
        if self.clusters:
            self.local_task_queue.put((self.solve_cluster, (self.clusters[0],), {}))

    def solve_cluster(self, cluster):
        print("Solving Cluster")
        # Apply TSP solver to the cluster
        # For example, let's assume a dummy solver that returns the cluster as-is
        solution = cluster

        # Add the solved cluster to cluster_solutions
        self.cluster_solutions.append(solution)

        # Check if there are more clusters to solve
        if len(self.cluster_solutions) < len(self.clusters):
            next_cluster = self.clusters[len(self.cluster_solutions)]
            self.local_task_queue.put((self.solve_cluster, (next_cluster,), {}))
        else:
            # All clusters are solved, enqueue task to connect them
            self.local_task_queue.put((self.connect_clusters, (), {}))

    def assemble_final_path(self, cluster_solutions, connections):
        print("Assembling final path", cluster_solutions, connections)
        final_path = []
        included_clusters = set()

        # Start with the first cluster's solution
        current_cluster_idx = 0
        final_path.extend(cluster_solutions[current_cluster_idx])
        included_clusters.add(current_cluster_idx)

        while len(included_clusters) < len(cluster_solutions):
            last_point = final_path[-1]
            closest_cluster_idx = None
            min_distance = float('inf')
            connection_point = None

            for i, cluster_path in enumerate(cluster_solutions):
                if i not in included_clusters:
                    for point_a, point_b in connections:
                        if point_a in cluster_solutions[current_cluster_idx] and point_b in cluster_path:
                            distance = self.graph[point_a][point_b]
                            if distance < min_distance:
                                min_distance = distance
                                closest_cluster_idx = i
                                connection_point = point_b

            if closest_cluster_idx is not None:
                # Add only the connection point from the next cluster to avoid duplicates
                final_path.append(connection_point)
                # Append the rest of the next cluster excluding the connection point
                next_cluster_path = [point for point in cluster_solutions[closest_cluster_idx] if
                                     point != connection_point]
                final_path.extend(next_cluster_path)

                included_clusters.add(closest_cluster_idx)
                current_cluster_idx = closest_cluster_idx
            else:
                # No more connectable clusters found
                break

        return final_path

    def connect_clusters(self):
        # Connect clusters and create final solution
        connections = self.find_closest_edges_between_clusters()
        final_solution = self.assemble_final_path(self.cluster_solutions, connections) + [self.points[0]]

        # Assuming you have a method to display the final solution
        print("Final Solution", final_solution)
        self.update_solution(final_solution)
        return True  # Worker stop


if __name__ == "__main__":
    pass
    #try:
    #    NeighborhoodSalesmanNetwork(10)
    #except KeyboardInterrupt:
    #    print("Continuing with next solution ...")


class RounderBouter(TSPBase):
    def __init__(self, num_points: int = 10):
        super().__init__(num_points, title_text_add_in=" Connect outer Points")

    @staticmethod
    def calculate_angle_and_distance_from_center(point, center):
        dx = point.x - center[0]
        dy = point.y - center[1]
        angle = math.atan2(dy, dx)
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return angle, distance

    def create_initial_task(self):
        return self.solve, (), {}

    def solve(self):
        # Calculate the centroid as the reference point
        centroid = (sum(p.x for p in self.points) / len(self.points),
                    sum(p.y for p in self.points) / len(self.points))

        # Sort points by angle and then by distance from the centroid
        sorted_points = sorted(self.points, key=lambda p: self.calculate_angle_and_distance_from_center(p, centroid))
        self.update_solution(sorted_points)
        self.local_solutions_checked += 1
        return True


if __name__ == "__main__":
    try:
        RounderBouter(10)
    except KeyboardInterrupt:
        print("Continuing with next solution ...")
