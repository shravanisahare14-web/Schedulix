import pandas as pd
from collections import defaultdict, deque


class DependencyGraph:

    def __init__(self, tasks_df):

        self.tasks_df = tasks_df

        self.graph = defaultdict(list)

        self.in_degree = defaultdict(int)

        self.tasks = {}

        self.build_graph()

    def build_graph(self):

        for _, row in self.tasks_df.iterrows():

            task_id = row["task_id"]

            self.tasks[task_id] = row.to_dict()

            if task_id not in self.in_degree:
                self.in_degree[task_id] = 0

        for _, row in self.tasks_df.iterrows():

            task_id = row["task_id"]

            dependencies = str(row["dependencies"]).strip()

            if dependencies and dependencies != "nan":

                dependency_list = [
                    dep.strip()
                    for dep in dependencies.split("|")
                ]

                for dep in dependency_list:

                    self.graph[dep].append(task_id)

                    self.in_degree[task_id] += 1

    def detect_cycle(self):

        queue = deque()

        in_degree_copy = self.in_degree.copy()

        for task in in_degree_copy:

            if in_degree_copy[task] == 0:
                queue.append(task)

        visited = 0

        while queue:

            current = queue.popleft()

            visited += 1

            for neighbor in self.graph[current]:

                in_degree_copy[neighbor] -= 1

                if in_degree_copy[neighbor] == 0:
                    queue.append(neighbor)

        return visited != len(self.tasks)

    def topological_sort(self):

        if self.detect_cycle():

            raise Exception(
                "Cycle detected in task dependencies."
            )

        queue = deque()

        in_degree_copy = self.in_degree.copy()

        for task in in_degree_copy:

            if in_degree_copy[task] == 0:
                queue.append(task)

        execution_order = []

        while queue:

            current = queue.popleft()

            execution_order.append(current)

            for neighbor in self.graph[current]:

                in_degree_copy[neighbor] -= 1

                if in_degree_copy[neighbor] == 0:
                    queue.append(neighbor)

        return execution_order


def load_dependency_graph():

    tasks_df = pd.read_csv("data/tasks.csv")

    graph = DependencyGraph(tasks_df)

    return graph


if __name__ == "__main__":

    graph = load_dependency_graph()

    order = graph.topological_sort()

    print("\nValid Execution Order:\n")

    for i, task in enumerate(order, start=1):

        print(f"{i}. {task}")