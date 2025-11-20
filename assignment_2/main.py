import itertools
import os

import numpy as np
import pandas as pd
import simpy


def load_data(file_path) -> pd.DataFrame:
    """Load data from a given file path."""

    # get relative path
    file_path = os.path.join(os.path.dirname(__file__), file_path)

    data = pd.read_csv(file_path)
    data = data.drop(
        columns=["Europe Flights", "Intercontinental Flights", "Total Flights"]
    )
    return data


def get_one_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Return a single month within the dataframe."""
    return df[df["Month"] == month]


def get_arrival_rate(df: pd.DataFrame, month: str, lanes: int = 50) -> float:
    """Calculates per minute arrival rate for a single lane if n is not specified.
    Assumes 16 hours of operation per day and 30 days in a month."""

    one_month = get_one_month(df, month)
    mean_arrival = one_month["Total Passengers"].mean()

    # 16h, 30 days, per minute
    arrival_rate = mean_arrival / (16 * 30 * 60)

    # per lane
    arrival_rate /= lanes

    return arrival_rate


class SecurityLane:

    def __init__(self, env: simpy.Environment, num_servers: int):
        self.env = env
        self.server = simpy.Resource(env, num_servers)
        self.service_time_mean = 1.0
        self.service_time_standard_dev = 0.25

    def service_passenger(self):
        # draw from normal distribution
        service_time = np.random.normal(
            loc=self.service_time_mean, scale=self.service_time_standard_dev
        )
        yield self.env.timeout(service_time)


def passenger(env: simpy.Environment, name: str, security_lane: SecurityLane):
    """A single passsenger passing through the security lane-"""

    # set timer for arrival
    arrival_time = env.now
    print(f"{name} arrives at {arrival_time:.2f}")
    with security_lane.server.request() as request:
        yield request
        yield env.process(security_lane.service_passenger())
        # set timer for leaving and calculate wait time
        # TODO: we need to collect these wait times here for analysis later
        leave_time = env.now
        wait_time = leave_time - arrival_time
        print(f"{name} waited for {wait_time:.2f} minutes")


def setup(env: simpy.Environment, num_machines: int, arrival_rate: float):
    """Setting up the security lane simulation"""

    # create security lane
    security_lane = SecurityLane(env, num_machines)
    passenger_count = itertools.count()
    init_passengers = 5

    for _ in range(init_passengers):
        env.process(passenger(env, f"Passenger {next(passenger_count)}", security_lane))

    # add more passengers while running
    # TODO: this needs to have a stopping condition, i.e. number of total passengers
    while True:
        yield env.timeout(1 / arrival_rate)
        passenger_id = next(passenger_count)
        env.process(passenger(env, f"Passenger {passenger_id}", security_lane))


if __name__ == "__main__":

    df = load_data("airport.csv")
    arrival_rate = get_arrival_rate(df, "September")

    print("------SECURITY LANE SIMULATION------")
    env = simpy.Environment()
    env.process(setup(env, num_machines=1, arrival_rate=arrival_rate))
    env.run(until=10)
