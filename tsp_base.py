# ------------------------------------------------------
# Uses TSPVisualizer to create a basis for every TSP-
# Solution that can easily run the gui and algorithm at the same time
# ------------------------------------------------------
import threading
import time
from queue import Queue
from timid_timer import TimidTimer
from random import randint
from common import calculate_num_solution, Point, one_tree_lower_bound, create_graph
import multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
    from gui import TSPVisualizer
import itertools
import sys


class TSPBase:
    def __init__(self, num_points: int = 10, title_text_add_in: str = "", gui_start: bool = True):
        self.visualizer = TSPVisualizer([], self.stop, self.quit, self.input_callback, num_points)
        self.points = [Point()] * num_points
        self.graph = self.better_lower_bound = None

        self.worker_process = None

        self._running = multiprocessing.Value('i', 0)  # Shared flag for running status
        self.requested = None  # So that it doesn't terminate before having started
        self.loop_active = True  # To keep the loop running
        self.event_loop_thread = None

        self.regenerate_points(None, False)

        self.visualizer.add_dynamic_text("TSP Solver" + title_text_add_in, (0, 0))
        self.solutions_checked_text = self.visualizer.add_dynamic_text(f"Solutions checked: 0/0 for "
                                                                       f"{len(self.points)} points", (0, 16))
        self.curr_length_text = self.visualizer.add_dynamic_text("Current best length: 0px", (0, 28))
        self.weight_text = self.visualizer.add_dynamic_text("Lower Bound weight + 20%: 0px", (0, 40))
        self.time_passed_text = self.visualizer.add_dynamic_text("Time passed: 00:00:00.000000", (0, 54))
        self.run_button = self.visualizer.add_button((10, 80, 40, 20), "Run", self.toggle_run)
        self.visualizer.add_button((60, 80, 120, 20), "Regenerate Points", self.regenerate_points)
        self.visualizer.add_button((10, 110, 170, 20), "Continue to next solution", self.next_solution)
        self.visualizer.add_button((30, 140, 130, 20), "Quit to desktop", self.visualizer.hard_stop)

        self._current_length = multiprocessing.Value('d', float('inf'))
        self._current_path = multiprocessing.Array('d', (num_points + 1) * 2)
        self._solutions_checked = multiprocessing.Value('i', 0)

        # Initialize the child's local variables with placeholders
        self.local_task_queue = None
        self.local_best_solution = []
        self.local_best_length = float('inf')
        self.local_solutions_checked = None

        if gui_start:
            self.start_gui()

    @property
    def current_path(self):
        with self._current_path.get_lock():
            return [Point(self._current_path[i], self._current_path[i + 1])
                    for i in range(0, len(self._current_path), 2)]

    @current_path.setter
    def current_path(self, value):
        flat_value = []
        for point in value:
            flat_value.extend([point.x, point.y])

        with self._current_path.get_lock():
            if len(flat_value) <= len(self._current_path):
                self._current_path[:len(flat_value)] = flat_value
                # Fill the rest with zeros
                self._current_path[len(flat_value):] = [0] * (len(self._current_path) - len(flat_value))
            else:
                raise ValueError("New path is too long for the allocated array.")

    @property
    def current_length(self):
        with self._current_length.get_lock():
            return self._current_length.value

    @current_length.setter
    def current_length(self, value):
        with self._current_length.get_lock():
            self._current_length.value = value

    @property
    def solutions_checked(self):
        with self._solutions_checked.get_lock():
            return self._solutions_checked.value

    @solutions_checked.setter
    def solutions_checked(self, value):
        with self._solutions_checked.get_lock():
            self._solutions_checked.value = value

    @property
    def running(self):
        with self._running.get_lock():
            return self._running.value

    @running.setter
    def running(self, value):
        with self._running.get_lock():
            self._running.value = value

    def start_gui(self):
        """
        Starts the event loop for TSPBase in a separate thread.
        Afterward, it starts the visualizer in the main thread
        """
        self.start_event_loop()
        self.visualizer.run()

    def start_event_loop(self):
        """Start the event loop for TSPBase in a separate thread"""
        if self.event_loop_thread is None or not self.event_loop_thread.is_alive():
            self.event_loop_thread = threading.Thread(target=self.event_loop)
            self.event_loop_thread.start()

    @staticmethod
    def format(number):
        """Format the number to use dot as a thousands separator."""
        return f"{number:,}".replace(",", ".")

    def event_loop(self):
        try:
            while self.loop_active:
                if self.requested and self.requested is not None and not self.running:
                    self.start()
                elif not self.requested and self.requested is not None or self.running == -1:
                    if self.running == -1:
                        self.toggle_run(self.run_button)
                    self.stop()
                timer = TimidTimer()
                while self.running == 1 and self.requested:
                    # Perform necessary TSP algorithm tasks
                    # 1. Update the texts
                    self.solutions_checked_text.update_text(f"Solutions checked: {self.format(self.solutions_checked)}/"
                                                            f"{self.format(calculate_num_solution(len(self.points)))}"
                                                            f" for {len(self.points)} points")
                    self.curr_length_text.update_text(f"Current best length: {round(self.current_length, 4)}px")
                    self.weight_text.update_text(f"Lower Bound weight + 20%: {round(((self.better_lower_bound // 10) * 2
                                                                                     + self.better_lower_bound), 4)}px")
                    self.time_passed_text.update_text(f"Time passed: {timer.tick()}")
                    # 2. Set the current path
                    self.visualizer.set_current_path(self.current_path)
                # Re-draw everything after worker stop to ensure data integrity
                if self.running == -1:
                    self.solutions_checked_text.update_text(f"Solutions checked: {self.format(self.solutions_checked)}/"
                                                            f"{self.format(calculate_num_solution(len(self.points)))}"
                                                            f" for {len(self.points)} points")
                    self.curr_length_text.update_text(f"Current best length: {round(self.current_length, 4)}px")
                    self.weight_text.update_text(f"Lower Bound weight + 20%: {round(((self.better_lower_bound // 10) * 2
                                                                                     + self.better_lower_bound), 4)}px")
                    self.time_passed_text.update_text(f"Time passed: {timer.tick()} (-0.1 seconds for last gui update)")
                    self.visualizer.set_current_path(self.current_path)
                # Reset the loop values for the next iteration/start
                self.solutions_checked = 0
                self.current_path = ()
                self.current_length = float('inf')
        except Exception as e:
            print(f"Exception in event_loop: {e}")

    def toggle_run(self, button):
        """Changes the text of the passed button and the internal requested variable to indicate a stop."""
        button.text = "Run" if button.text == "Stop" else "Stop"
        self.requested = not self.requested

    def close(self):
        """Stops TSPBase completely as well as soft-stopping the visualizer."""
        self.stop()
        self.loop_active = False
        if self.visualizer.running:
            self.visualizer.stop()

    def start(self, debug_mode: bool = False):
        """
        Starts the solution (worker) in a separate thread or process.

        Arguments:
            debug_mode -- If true, it starts the worker in a new thread instead of a new process so prints work.
        """
        self.current_length = sum([self.graph[point][self.points[(i + 1) % len(self.points)]]
                                   for i, point in enumerate(self.points)])
        self.current_path = self.points + [self.points[0]]
        self.visualizer.set_current_path(self.points)

        self.running = 1
        if not debug_mode:
            self.worker_process = multiprocessing.Process(target=self.worker, args=())
        else:
            self.worker_process = threading.Thread(target=self.worker, args=())
        self.worker_process.start()

    def stop(self):
        """Stops TSPBase, if running or not and joins the worker thread or process."""
        self.running = 0
        self.requested = None

        if self.worker_process:
            self.worker_process.join()
            self.worker_process = None

    def create_initial_task(self):
        """Create and return the initial task data"""
        return ()

    def worker(self, num_update_interval: int = 1000, sol_update_interval: int = 10000):
        self.local_task_queue = Queue()
        self.local_best_solution = list(self.current_path)
        self.local_best_length = float(self.current_length)
        self.local_solutions_checked = 0

        # Add the initial task
        initial_task = self.create_initial_task()
        self.local_task_queue.put(initial_task)

        iteration_count = 0

        while self.running:
            if not self.local_task_queue.empty():
                task = self.local_task_queue.get()
                if self.process_task(task):
                    # Task was final
                    self.current_path = self.local_best_solution
                    self.current_length = self.local_best_length
                    self.solutions_checked = self.local_solutions_checked
                    time.sleep(0.1)  # So the gui text can get updated
                    self.running = -1  # Known as worker stop
                iteration_count += 1  # Move to outer loop?
                if iteration_count % num_update_interval == 0:
                    self.solutions_checked = self.local_solutions_checked
                if iteration_count >= sol_update_interval:
                    self.current_path = self.local_best_solution
                    self.current_length = self.local_best_length
                    iteration_count = 0
            else:
                time.sleep(0.01)

    def is_final_solution(self):
        """Returns true if the task queue is empty"""
        if self.local_task_queue.empty():
            return True
        return False

    def process_task(self, task: tuple) -> bool:
        """Splits a task tuple into it's components and executes them."""
        function, args, kwargs = task
        return function(*args, **kwargs)

    def k_opt(self, route: list, k: int = 2) -> list:
        if k < 2 or k > len(route):
            raise ValueError("Invalid value for k")

        best_route = route
        improved = True

        while improved:
            improved = False
            for indices in itertools.combinations(range(len(route)), k):
                for new_edges in itertools.permutations(indices):
                    if new_edges != indices:
                        new_route = self.swap_edges(best_route, indices, new_edges)
                        if self.route_cost(new_route) < self.route_cost(best_route):
                            best_route = new_route
                            improved = True

        return best_route

    @staticmethod
    def swap_edges(route, old_edges, new_edges):
        # This is a simplified example of swapping edges, further optimization may be needed
        new_route = route[:]
        # Assuming direct swap of points for simplicity
        for i, j in zip(old_edges, new_edges):
            new_route[i] = route[j]
        return new_route

    @staticmethod
    def route_cost(route):
        cost = 0
        for i in range(len(route)):
            cost += route[i].distance_to(route[(i + 1) % len(route)])
        return cost

    def update_solution(self, new_solution: list, new_length: int = None, update_last: bool = True):
        """
        Takes care of all the stuff needed to properly update the solution.

        Arguments:
            new_solution -- a list of Point objects
            new_length -- optional already calculated length of new_solution, if not passed it will be calculated
            update_last -- if the first point should be appended as the last to properly display it.
        """
        if not new_length:
            new_length = 0
            for i in range(len(new_solution) - 1):
                new_length += self.graph[new_solution[i]][new_solution[(i + 1) % len(self.points)]]
            new_length += self.graph[new_solution[-1]][new_solution[0]]
        if new_length < self.local_best_length:
            if not new_solution[-1] == new_solution[0] and update_last:
                self.local_best_solution = new_solution + [new_solution[0]]
            else:
                self.local_best_solution = new_solution
            self.local_best_length = new_length
            # self.visualizer.set_current_path(new_solution.copy())

    def input_callback(self):
        inp = int(self.visualizer.input)
        if 2 < inp < 100:
            self.points = [Point()] * inp
            self._current_path = multiprocessing.Array('d', (len(self.points) + 1) * 2)
            self.solutions_checked_text.update_text(f"Solutions checked: 0/0 for {len(self.points)} points")
            self.regenerate_points(None, True)
        else:
            self.solutions_checked_text.update_text(f"Solutions checked: 0/0 for {inp} points (non-start able)")
            if self.running:
                self.toggle_run(self.run_button)

    def regenerate_points(self, _, reset_text: bool = True):
        """
        Replace all current points with new ones randomly generated with the same length as the old points list,
        it also stops the worker if it's currently running.

        Arguments:
            _ -- Unused button argument
            reset_text -- If set to true, it resets the text to display 0 for all stats.
        """
        self.points = self.generate_random_points_2d((self.visualizer.width, self.visualizer.height), len(self.points))
        self.visualizer.replace_points(self.points.copy())
        if self.running:
            self.toggle_run(self.run_button)
            # self.run_button.text = "Start"
            # self.running = 0
        if self.requested:
            self.requested = None
        self.graph = create_graph(self.points.copy())
        self.better_lower_bound = round(one_tree_lower_bound(self.graph, self.points.copy()), 2)
        if reset_text:  # Reset the text
            self.solutions_checked_text.update_text(f"Solutions checked: 0/0 for {len(self.points)} points")
            self.curr_length_text.update_text("Current best length: 0px")
            self.weight_text.update_text("Lower Bound weight + 20%: 0px")
            self.time_passed_text.update_text("Time passed: 00:00:00.000000")

    def next_solution(self, _):
        """Stops the current TSPBase object and raises a keyboard interrupt."""
        self.loop_active = False
        if self.event_loop_thread:
            self.event_loop_thread.join()
            self.event_loop_thread = None
        raise KeyboardInterrupt

    def quit(self, _=None):
        """Stops the current TSPBase object, hard stops the visualizer and exits using sys.exit(0)."""
        self.close()
        self.loop_active = False
        sys.exit(0)

    @staticmethod
    def generate_random_points_2d(dimensions: tuple, amount: int = 1):
        points = []
        for i in range(amount):
            points.append(Point(randint(0, dimensions[0] - 1), randint(0, dimensions[1] - 1)))
        return points

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle any thread, multiprocessing or pygame objects
        unpickable_objects = ["visualizer", "run_button",
                              "solutions_checked_text", "curr_length_text", "time_passed_text", "weight_text",
                              "event_loop_thread", "manager", "process_pool",
                              "solutions_checked"]  # , "solution_lock", "task_queue"]
        for obj in unpickable_objects:
            if obj in state:
                del state[obj]
        return state

    def __get_state__(self):
        state = {}
        needed_objects = ["_running", "_solutions_checked",
                          "_solution", "points", "graph",
                          "_solution_lock", "better_lower_bound"]

        for obj in needed_objects:
            state[obj] = self.__dict__.get(obj)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add anything back that doesn't exist in the pickle and is needed from the subprocess
        pass  # In other words nothing so far


class TSPTest(TSPBase):
    def process_task(self, task):
        self.update_solution(self.points.copy())


if __name__ == "__main__":
    TSPTest()
