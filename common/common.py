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

    def distance_to(self, other: "Point") -> float:
        """
        Calculate Euclidean/Pythagorean distance between two Point instances, using the standard math module.

        Arguments:
            other -- a Point instance
        """
        if not isinstance(other, Point):
            raise NotImplementedError("This function only supports Point objects.")
        return math.dist([self.x, self.y], [other.x, other.y])

    def __lt__(self, other: "Point"):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def __eq__(self, other: "Point") -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Point(x: {self.x}, y: {self.y})"


@dataclass
class AdvPoint:
    x: int = 0
    y: int = 0
    z: int = None
    v: int = None
    floating: bool = False

    def distance_to(self, other: "AdvPoint") -> float:
        """
        Calculate Euclidean/Pythagorean distance between two Point instances, using the standard math module.

        Arguments:
            other -- a Point instance
        """
        if not isinstance(other, AdvPoint):
            raise TypeError("This function only supports Point objects.")

        self_coords = [self.x, self.y] + ([self.z] if self.z is not None else []) + (
            [self.v] if self.v is not None else [])
        other_coords = [other.x, other.y] + ([other.z] if other.z is not None else []) + (
            [other.v] if other.v is not None else [])

        return math.dist(self_coords, other_coords)

    def __lt__(self, other: "AdvPoint"):
        if not isinstance(other, AdvPoint):
            return NotImplemented

        self_vals = (
        self.x, self.y, self.z if self.z is not None else float('inf'), self.v if self.v is not None else float('inf'))
        other_vals = (other.x, other.y, other.z if other.z is not None else float('inf'),
                      other.v if other.v is not None else float('inf'))

        return self_vals < other_vals

    def __eq__(self, other: "AdvPoint") -> bool:
        if not isinstance(other, AdvPoint):
            return NotImplemented

        return (self.x, self.y, self.z, self.v) == (other.x, other.y, other.z, other.v)

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.v))

    def __getitem__(self, key):
        attributes = ('x', 'y', 'z', 'v')
        if 0 <= key < len(attributes):
            value = getattr(self, attributes[key])
            if value is not None:
                return value
            else:
                raise IndexError(f"Attribute '{attributes[key]}' is not set")
        else:
            raise IndexError("Point index out of range")

    def __repr__(self) -> str:
        cls_name = "Point(" if not self.floating else "FloatingPoint("
        parts = [f"x: {self.x}", f"y: {self.y}"]
        if self.z is not None:
            parts.append(f"z: {self.z}")
        if self.v is not None:
            parts.append(f"v: {self.v}")
        return cls_name + ", ".join(parts) + ")"


def calculate_num_solution(n: int = 8) -> int:  # Calculates the number of unique solutions
    """
    Calculate the number of unique solutions for a given TSP length. The calculation is fac(n-1) // 2.

    Arguments:
        n -- number of points for the TSP
    """
    return math.factorial(n-1) // 2


def create_graph(points: list) -> dict:  # Creates a data-structure that has the distance of any point to any point
    """
    Creates a dictionary of dictionaries that contains the distance from all points to all points except itself.

    Arguments:
        points -- an iterator with Point instances or something similar
    """
    graph = {}
    for point in points:
        graph[point] = {other: point.distance_to(other) for other in points if other != point}
    return graph


def prims_algorithm(graph: dict, start_vertex: Point) -> list:
    """
    Standard prims algorithm, connect the lowest cost edge reachable, it's an MST algorithm.

    Arguments:
        graph -- a graph created using the create_graph function or similar
        start_vertex -- the start point of the algorithm
    """
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


def one_tree_lower_bound(graph: dict, points: list) -> float:
    """
    Uses the function prims_algorithm to create all possible one trees and then chooses the highest cost one.

    Arguments:
        graph -- a graph created using the create_graph function or similar
        points -- the points used for the graph
    """
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
