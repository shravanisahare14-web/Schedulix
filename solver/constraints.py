import pandas as pd


class ConstraintManager:

    def __init__(self):

        self.tasks_df = pd.read_csv("data/tasks.csv")

        self.resources_df = pd.read_csv("data/resources.csv")

        self.resource_capacity = {}

        self.resource_skills = {}

        self.resource_working_hours = {}

        self.initialize_resources()

    def initialize_resources(self):

        for _, row in self.resources_df.iterrows():

            resource_id = row["resource_id"]

            self.resource_capacity[resource_id] = int(
                row["capacity_hours"]
            )

            self.resource_skills[resource_id] = set(
                str(row["skills"]).split("|")
            )

            # Default working day
            self.resource_working_hours[resource_id] = {

                "start": 9,
                "end": 17

            }

    # --------------------------------------------------
    # SKILL CONSTRAINT
    # --------------------------------------------------

    def validate_skill(
        self,
        resource_id,
        required_skill
    ):

        return required_skill in self.resource_skills[
            resource_id
        ]

    # --------------------------------------------------
    # CAPACITY CONSTRAINT
    # --------------------------------------------------

    def validate_capacity(
        self,
        resource_id,
        current_hours,
        task_duration
    ):

        capacity = self.resource_capacity[
            resource_id
        ]

        return (
            current_hours + task_duration
            <= capacity
        )

    # --------------------------------------------------
    # WORKING HOURS CONSTRAINT
    # --------------------------------------------------

    def validate_working_hours(
        self,
        resource_id,
        start_time,
        duration
    ):

        shift_start = self.resource_working_hours[
            resource_id
        ]["start"]

        shift_end = self.resource_working_hours[
            resource_id
        ]["end"]

        finish_time = start_time + duration

        return (
            shift_start
            <= start_time
            and
            finish_time <= shift_end
        )

    # --------------------------------------------------
    # DEPENDENCY CONSTRAINT
    # --------------------------------------------------

    def validate_dependencies(
        self,
        task_id,
        dependencies,
        completed_tasks
    ):

        if (
            dependencies is None
            or
            str(dependencies).lower() == "nan"
            or
            str(dependencies).strip() == ""
        ):

            return True

        dependency_list = [

            dep.strip()

            for dep in str(
                dependencies
            ).split("|")

        ]

        for dependency in dependency_list:

            if dependency not in completed_tasks:

                return False

        return True

    # --------------------------------------------------
    # RESOURCE AVAILABILITY
    # --------------------------------------------------

    def find_eligible_resources(
        self,
        required_skill
    ):

        eligible = []

        for resource_id, skills in self.resource_skills.items():

            if required_skill in skills:

                eligible.append(resource_id)

        return eligible

    # --------------------------------------------------
    # RESOURCE SUMMARY
    # --------------------------------------------------

    def get_resource_summary(self):

        summary = []

        for resource_id in self.resource_capacity:

            summary.append({

                "resource_id": resource_id,

                "skills": list(
                    self.resource_skills[resource_id]
                ),

                "capacity": self.resource_capacity[
                    resource_id
                ]

            })

        return summary


if __name__ == "__main__":

    manager = ConstraintManager()

    print("\nRESOURCE SUMMARY\n")

    for item in manager.get_resource_summary():

        print(item)