# ------------------------------------------------------
# Provides datatypes and functions that are used through-
# out the project and don't belong to a specific type
# ------------------------------------------------------
from dataclasses import dataclass
import math
import heapq


@dataclass
class Point:
    x: int = 0
    y: int = 0

    def distance_to(self, other) -> float:
        return math.dist([self.x, self.y], [other.x, other.y])

    def __lt__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Point(x: {self.x}, y: {self.y})"


def calculate_num_solution(n: int = 8):  # Calculates the number of unique solutions
    return math.factorial(n-1) // 2


def create_graph(points):  # Creates a data-structure that has the distance of any point to any point
    graph = {}
    for point in points:
        graph[point] = {other: point.distance_to(other) for other in points if other != point}
    return graph


def prims_algorithm(graph, start_vertex):
    mst = []  # Works by finding the smallest cost edge connected to the current route
    visited = {start_vertex}
    edges = [(cost, start_vertex, to) for to, cost in graph[start_vertex].items()]
    heapq.heapify(edges)  # Orders them after cost (lowest is at the bottom)

    while edges:
        cost, from_vertex, to_vertex = heapq.heappop(edges)  # Get the lowest cost edge
        if to_vertex not in visited:
            visited.add(to_vertex)
            mst.append((from_vertex, to_vertex, cost))

            for next_vertex, next_cost in graph[to_vertex].items():
                if next_vertex not in visited:
                    heapq.heappush(edges, (next_cost, to_vertex, next_vertex))

    return mst


def one_tree_lower_bound(graph, points):
    highest_cost = 0

    for point in points:
        # Create a graph without the current point and also remove the point from neighbors
        reduced_graph = {}
        for p, neighbors in graph.items():
            if p != point:  # Don't add point
                reduced_graph[p] = {other: dist for other, dist in neighbors.items() if other != point}
                # Don't add any mentions of point either
        # Find MST of the reduced graph
        start_vertex = next(iter(reduced_graph))  # Fetches the first key from a dictionary
        mst = prims_algorithm(reduced_graph, start_vertex)

        # Find two shortest edges from 'point' to the vertices in the MST
        shortest_edges = sorted([(point.distance_to(other), point, other) for other in reduced_graph],
                                key=lambda x: x[0])[:2]

        # Calculate the cost of the 1-tree
        tree_cost = sum(cost for _, _, cost in mst) + sum(edge[0] for edge in shortest_edges)

        # Update the highest cost if necessary
        if tree_cost > highest_cost:
            highest_cost = tree_cost

    return highest_cost


if __name__ == "__main__":
    # Example usage
    route = [Point(0, 0), Point(1, 2), Point(3, 4), Point(5, 6)]

    # Create a complete graph from the list of points
    graph_test = create_graph(route)

    # Find the minimum spanning tree
    mst_test = prims_algorithm(graph_test, route[0])
    print("Minimum Spanning Tree:", mst_test)

    # Calculate the 1-tree lower bound for TSP
    lower_bound = one_tree_lower_bound(graph_test, route)
    print("1-Tree Lower Bound for TSP:", lower_bound)
