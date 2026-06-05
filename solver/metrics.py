import pandas as pd


class MetricsCalculator:

    def __init__(self, schedule):

        self.schedule = pd.DataFrame(schedule)

    # -----------------------------------
    # TOTAL TASKS
    # -----------------------------------

    def total_tasks(self):

        return len(self.schedule)

    # -----------------------------------
    # ON-TIME TASKS
    # -----------------------------------

    def on_time_tasks(self):

        return len(

            self.schedule[

                self.schedule["finish_time"]

                <=

                self.schedule["deadline"]

            ]

        )

    # -----------------------------------
    # LATE TASKS
    # -----------------------------------

    def late_tasks(self):

        return len(

            self.schedule[

                self.schedule["finish_time"]

                >

                self.schedule["deadline"]

            ]

        )

    # -----------------------------------
    # TOTAL LATENESS
    # -----------------------------------

    def total_lateness(self):

        lateness = (

            self.schedule["finish_time"]

            -

            self.schedule["deadline"]

        )

        lateness = lateness.clip(lower=0)

        return int(lateness.sum())

    # -----------------------------------
    # ON-TIME PERCENTAGE
    # -----------------------------------

    def on_time_percentage(self):

        total = self.total_tasks()

        if total == 0:

            return 0

        return round(

            (

                self.on_time_tasks()

                /

                total

            )

            * 100,

            2

        )

    # -----------------------------------
    # RESOURCE UTILIZATION
    # -----------------------------------

    def resource_utilization(self):

        utilization = {}

        grouped = self.schedule.groupby(

            "resource_id"

        )

        for resource, group in grouped:

            total_hours = (

                group["finish_time"]

                -

                group["start_time"]

            ).sum()

            utilization[resource] = round(

                total_hours,

                2

            )

        return utilization

    # -----------------------------------
    # AVG TASK DURATION
    # -----------------------------------

    def average_task_duration(self):

        durations = (

            self.schedule["finish_time"]

            -

            self.schedule["start_time"]

        )

        return round(

            durations.mean(),

            2

        )

    # -----------------------------------
    # MAX TASK DURATION
    # -----------------------------------

    def max_task_duration(self):

        durations = (

            self.schedule["finish_time"]

            -

            self.schedule["start_time"]

        )

        return int(

            durations.max()

        )

    # -----------------------------------
    # SUMMARY
    # -----------------------------------

    def summary(self):

        return {

            "total_tasks":

                self.total_tasks(),

            "on_time_tasks":

                self.on_time_tasks(),

            "late_tasks":

                self.late_tasks(),

            "total_lateness":

                self.total_lateness(),

            "on_time_percentage":

                self.on_time_percentage(),

            "average_task_duration":

                self.average_task_duration(),

            "max_task_duration":

                self.max_task_duration(),

            "resource_utilization":

                self.resource_utilization()

        }


if __name__ == "__main__":

    sample_schedule = [

        {

            "task_id": "T1",

            "resource_id": "R1",

            "start_time": 0,

            "finish_time": 4,

            "deadline": 8

        },

        {

            "task_id": "T2",

            "resource_id": "R2",

            "start_time": 5,

            "finish_time": 18,

            "deadline": 16

        }

    ]

    metrics = MetricsCalculator(

        sample_schedule

    )

    print(

        metrics.summary()

    )