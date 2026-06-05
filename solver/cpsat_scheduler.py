import pandas as pd

from ortools.sat.python import cp_model

from dependency_graph import DependencyGraph


class CPSATScheduler:

    def __init__(self):

        self.tasks_df = pd.read_csv("data/tasks.csv")

        self.resources_df = pd.read_csv("data/resources.csv")

        self.model = cp_model.CpModel()

        self.schedule = []

        self.horizon = 100

    def solve(self):

        tasks = {

            row["task_id"]: row

            for _, row in self.tasks_df.iterrows()

        }

        resources = {

            row["resource_id"]: row

            for _, row in self.resources_df.iterrows()

        }

        start_vars = {}

        end_vars = {}

        resource_vars = {}

        interval_vars = {}

        # --------------------------------
        # CREATE VARIABLES
        # --------------------------------

        for task_id, task in tasks.items():

            duration = int(task["duration"])

            start_vars[task_id] = self.model.NewIntVar(

                0,
                self.horizon,
                f"start_{task_id}"

            )

            end_vars[task_id] = self.model.NewIntVar(

                0,
                self.horizon,
                f"end_{task_id}"

            )

            self.model.Add(

                end_vars[task_id]

                ==

                start_vars[task_id] + duration

            )

            interval_vars[task_id] = self.model.NewIntervalVar(

                start_vars[task_id],

                duration,

                end_vars[task_id],

                f"interval_{task_id}"

            )

        # --------------------------------
        # RESOURCE ASSIGNMENT VARIABLES
        # --------------------------------

        for task_id, task in tasks.items():

            eligible_resources = []

            required_skill = task["required_skill"]

            for resource_id, resource in resources.items():

                skills = str(
                    resource["skills"]
                ).split("|")

                if required_skill in skills:

                    eligible_resources.append(
                        resource_id
                    )

            resource_vars[task_id] = {

                resource_id:

                self.model.NewBoolVar(

                    f"{task_id}_{resource_id}"

                )

                for resource_id in eligible_resources

            }

            self.model.Add(

                sum(

                    resource_vars[task_id].values()

                )

                == 1

            )

        # --------------------------------
        # DEPENDENCY CONSTRAINTS
        # --------------------------------

        for task_id, task in tasks.items():

            dependencies = str(
                task["dependencies"]
            ).strip()

            if (

                dependencies

                and

                dependencies != "nan"

            ):

                dependency_list = [

                    dep.strip()

                    for dep in dependencies.split("|")

                ]

                for dep in dependency_list:

                    self.model.Add(

                        start_vars[task_id]

                        >=

                        end_vars[dep]

                    )

        # --------------------------------
        # RESOURCE CONFLICTS
        # --------------------------------

        for resource_id in resources:

            resource_intervals = []

            resource_presence = []

            for task_id in tasks:

                if (

                    resource_id

                    in

                    resource_vars[task_id]

                ):

                    presence = resource_vars[
                        task_id
                    ][resource_id]

                    duration = int(
                        tasks[task_id]["duration"]
                    )

                    optional_interval = self.model.NewOptionalIntervalVar(

                        start_vars[task_id],

                        duration,

                        end_vars[task_id],

                        presence,

                        f"{task_id}_{resource_id}_interval"

                    )

                    resource_intervals.append(
                        optional_interval
                    )

            self.model.AddNoOverlap(
                resource_intervals
            )

        # --------------------------------
        # LATENESS OBJECTIVE
        # --------------------------------

        lateness_vars = []

        for task_id, task in tasks.items():

            deadline = int(task["deadline"])

            lateness = self.model.NewIntVar(

                0,
                self.horizon,
                f"lateness_{task_id}"

            )

            self.model.Add(

                lateness

                >=

                end_vars[task_id] - deadline

            )

            lateness_vars.append(
                lateness
            )

        self.model.Minimize(

            sum(lateness_vars)

        )

        # --------------------------------
        # SOLVE
        # --------------------------------

        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = 10

        status = solver.Solve(self.model)

        if status not in (

            cp_model.OPTIMAL,

            cp_model.FEASIBLE

        ):

            return []

        # --------------------------------
        # EXTRACT RESULTS
        # --------------------------------

        for task_id in tasks:

            assigned_resource = None

            for resource_id in resource_vars[task_id]:

                if (

                    solver.Value(

                        resource_vars[task_id][resource_id]

                    )

                    == 1

                ):

                    assigned_resource = resource_id

                    break

            self.schedule.append({

                "task_id": task_id,

                "task_name": tasks[
                    task_id
                ]["task_name"],

                "resource_id": assigned_resource,

                "start_time": solver.Value(
                    start_vars[task_id]
                ),

                "finish_time": solver.Value(
                    end_vars[task_id]
                ),

                "deadline": int(
                    tasks[task_id]["deadline"]
                ),

                "priority": int(
                    tasks[task_id]["priority"]
                )

            })

        self.schedule.sort(

            key=lambda x: x["start_time"]

        )

        return self.schedule


if __name__ == "__main__":

    scheduler = CPSATScheduler()

    result = scheduler.solve()

    print("\nCP-SAT OPTIMIZED SCHEDULE\n")

    for task in result:

        print(task)