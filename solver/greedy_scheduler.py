import pandas as pd
import heapq

from dependency_graph import DependencyGraph


class GreedyScheduler:

    def __init__(self):

        self.tasks_df = pd.read_csv("data/tasks.csv")

        self.resources_df = pd.read_csv("data/resources.csv")

        self.schedule = []

        self.resource_load = {}

        self.task_finish_times = {}

        self.initialize_resources()

    def initialize_resources(self):

        for _, row in self.resources_df.iterrows():

            self.resource_load[row["resource_id"]] = 0

    def get_resource_for_skill(self, skill):

        eligible = []

        for _, row in self.resources_df.iterrows():

            skills = str(row["skills"]).split("|")

            if skill in skills:

                heapq.heappush(
                    eligible,
                    (
                        self.resource_load[row["resource_id"]],
                        row["resource_id"]
                    )
                )

        if not eligible:
            return None

        _, resource_id = heapq.heappop(eligible)

        return resource_id

    def build_schedule(self):

        graph = DependencyGraph(self.tasks_df)

        execution_order = graph.topological_sort()

        task_lookup = {
            row["task_id"]: row
            for _, row in self.tasks_df.iterrows()
        }

        for task_id in execution_order:

            task = task_lookup[task_id]

            duration = int(task["duration"])

            deadline = int(task["deadline"])

            priority = int(task["priority"])

            skill = task["required_skill"]

            dependencies = str(
                task["dependencies"]
            ).strip()

            earliest_start = 0

            if dependencies and dependencies != "nan":

                dependency_list = [
                    dep.strip()
                    for dep in dependencies.split("|")
                ]

                earliest_start = max(
                    self.task_finish_times[dep]
                    for dep in dependency_list
                )

            resource_id = self.get_resource_for_skill(skill)

            if resource_id is None:

                continue

            resource_available = self.resource_load[
                resource_id
            ]

            start_time = max(
                earliest_start,
                resource_available
            )

            finish_time = start_time + duration

            lateness = max(
                0,
                finish_time - deadline
            )

            self.schedule.append({

                "task_id": task_id,

                "task_name": task["task_name"],

                "resource_id": resource_id,

                "start_time": start_time,

                "finish_time": finish_time,

                "deadline": deadline,

                "priority": priority,

                "lateness": lateness

            })

            self.resource_load[
                resource_id
            ] = finish_time

            self.task_finish_times[
                task_id
            ] = finish_time

        return self.schedule


if __name__ == "__main__":

    scheduler = GreedyScheduler()

    result = scheduler.build_schedule()

    print("\nGREEDY SCHEDULE\n")

    for task in result:

        print(task)