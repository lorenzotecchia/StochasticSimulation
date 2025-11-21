import itertools
import os
import numpy as np
import pandas as pd
import simpy
from tqdm import tqdm
from scipy import stats


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

    def __init__(
        self,
        env: simpy.Environment,
        num_servers: int,
        st_mean: float = 1.0,
        st_std: float = 0.25,
    ):
        self.env = env
        self.server = simpy.Resource(env, num_servers)
        self.service_time_mean = st_mean
        self.service_time_standard_dev = st_std

    def service_passenger(self):
        # draw from normal distribution
        service_time = np.random.normal(
            loc=self.service_time_mean, scale=self.service_time_standard_dev
        )
        # ensure non-negative service time
        service_time = max(0, service_time)
        yield self.env.timeout(service_time)


def passenger(
    env: simpy.Environment,
    name: str,
    security_lane: SecurityLane,
    waiting_times: list,
    verbose: bool = False,
):
    """A single passsenger passing through the security lane-"""

    # set timer for arrival

    with security_lane.server.request() as request:
        arrival_time = env.now
        if verbose:
            print(f"{name} arrives at {arrival_time:.2f}")
        yield request
        queue_time = env.now
        # service starts
        yield env.process(security_lane.service_passenger())
        # set timer for leaving and calculate wait time
        leave_time = env.now
        total_wait_time = leave_time - arrival_time
        wait_queue_time = queue_time - arrival_time
        waiting_times.append(wait_queue_time)
        if verbose:
            print(f"{name} waited for {wait_queue_time:.2f} minutes")


def setup(
    env: simpy.Environment,
    num_machines: int,
    arrival_rate: float,
    waiting_times: list,
    verbose: bool = False,
    passengers_passed: int = 1000,
):
    """Setting up the security lane simulation"""

    # create security lane
    security_lane = SecurityLane(env, num_machines)
    passenger_count = itertools.count()
    init_passengers = 0

    for _ in range(init_passengers):
        env.process(
            passenger(
                env,
                f"Passenger {next(passenger_count)}",
                security_lane,
                waiting_times,
                verbose=verbose,
            )
        )
    # add more passengers while running
    passed = 0
    while passed < passengers_passed:
        yield env.timeout(np.random.exponential(1 / arrival_rate))
        passenger_id = next(passenger_count)
        env.process(
            passenger(
                env,
                f"Passenger {passenger_id}",
                security_lane,
                waiting_times,
                verbose=verbose,
            )
        )
        passed += 1


def run_simulation(
    arrival_rate: float, verbose: bool = False, passengers_passed: int = 1000
) -> list[float]:
    """Runs a single simulation"""
    waiting_times = []
    env = simpy.Environment()
    env.process(
        setup(
            env,
            num_machines=1,
            arrival_rate=arrival_rate,
            waiting_times=waiting_times,
            verbose=verbose,
            passengers_passed=passengers_passed,
        )
    )
    env.run()
    return waiting_times


if __name__ == "__main__":

    df = load_data("airport.csv")
    arrival_rate = get_arrival_rate(df, "September")
    Q2A = True

    if Q2A:

        arrival_rate = 0.85
        R = 40
        waiting_times_collector = []

        for i in tqdm(range(R)):
            waiting_times = run_simulation(
                arrival_rate, passengers_passed=3000, verbose=True
            )
            waiting_times_collector.append(waiting_times)

        print("------SECURITY LANE SIMULATION------")
        print("------------Q2A  RESULTS------------")

        passengers_passed = [len(w) for w in waiting_times_collector]
        mean_waiting_time = np.mean([np.mean(ws) for ws in waiting_times_collector])
        print(f"Mean passengers passed: {np.mean(passengers_passed):.0f}")
        print(f"Mean waiting time: {mean_waiting_time:.2f} minutes")
