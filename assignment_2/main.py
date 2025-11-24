import itertools
import os
import numpy as np
import pandas as pd
import simpy
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt


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
        n_customers=3000,
    ):
        self.env = env
        self.server = simpy.Resource(env, num_servers)
        self.service_time_mean = st_mean
        self.service_time_standard_dev = st_std
        self.n_customers = n_customers

        self.service_times = stats.truncnorm.rvs(
            -self.service_time_mean / self.service_time_standard_dev,
            np.inf,
            loc=self.service_time_mean,
            scale=self.service_time_standard_dev,
            size=self.n_customers,
        )

        self.current_index = 0

    def service_passenger(self):
        # draw from normal distribution
        # ensure non-negative service time

        service_time = self.service_times[self.current_index]
        self.current_index += 1

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
    num_servers: int,
    arrival_rate: float,
    waiting_times: list,
    verbose: bool = False,
    st_std: float = 0.25,
    passengers_passed: int = 3000,
):
    """Setting up the security lane simulation"""

    # create security lane
    security_lane = SecurityLane(env, num_servers, st_std=st_std)
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
    arrival_rate: float,
    st_std: float = 0.25,
    verbose: bool = False,
    passengers_passed: int = 3000,
    num_servers: int = 1,
) -> list[float]:
    """Runs a single simulation"""
    waiting_times = []
    env = simpy.Environment()
    env.process(
        setup(
            env,
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            waiting_times=waiting_times,
            verbose=verbose,
            passengers_passed=passengers_passed,
            st_std=st_std,
        )
    )
    env.run()
    return waiting_times


def run_multiple_simulations(
    arrival_rate: float,
    st_std: float = 0.25,
    verbose: bool = False,
    passengers_passed: int = 3000,
    num_servers: int = 1,
    num_replications=40,
    warm_up: int = 0,
):
    waiting_times_collector: list = []

    for i in tqdm(range(num_replications)):
        waiting_times = run_simulation(
            arrival_rate,
            st_std=st_std,
            passengers_passed=passengers_passed,
            verbose=verbose,
            num_servers=num_servers,
        )
        waiting_times_collector.append(waiting_times)

    waiting_times_collector = np.array(waiting_times_collector)

    # passengesers passed per replication
    passengers_passed = [len(w) for w in waiting_times_collector]

    # discard warm-up period
    waiting_times_collector = waiting_times_collector[:, warm_up:]
    mean_waiting_time = np.mean([np.mean(ws) for ws in waiting_times_collector])
    # use bessels correction for std dev (ddof=1)
    std_waiting_time = np.std([np.mean(ws) for ws in waiting_times_collector], ddof=1)

    return (
        passengers_passed,
        mean_waiting_time,
        std_waiting_time,
        waiting_times_collector,
    )


def check_validity(
    average: float, std: float, length: int, theoretical_value: float, t_value: float
):

    t_stat = (average - theoretical_value) / (std / np.sqrt(length))
    return t_stat < t_value


def plot_waiting_times_cumavg(
    waiting_times_collector: list,
    reps_to_plot: int,
    warm_up: int = 0,
    save_path: str = "",
    show: bool = False,
):
    for r in range(reps_to_plot):
        cumavg = np.cumsum(waiting_times_collector[r]) / np.arange(
            1, len(waiting_times_collector[r]) + 1
        )
        plt.plot(cumavg, alpha=0.5)

    avg_per_customer = np.mean(waiting_times_collector, axis=0)
    cumavg_per_customer = np.cumsum(avg_per_customer) / np.arange(
        1, len(avg_per_customer) + 1
    )
    plt.plot(
        cumavg_per_customer, color="black", label="ensemble-averaged cumulative mean"
    )
    if warm_up:
        plt.axvline(x=warm_up, color="red", label="warm up", ls="--")

    plt.xlabel("customer index")
    plt.ylabel("cumulative average waiting time")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()

    if show:
        plt.show()

    # save if path is there
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":

    df = load_data("airport.csv")
    Q2A = True
    WARM_UP_SWEEP = False
    PLOT_WARM_UP_SWEEP = False
    Q2B = False

    if Q2A:
        utilization = 0.85
        service_time_mean = 1
        service_time_std = 0.25
        arrival_rate = utilization / service_time_mean

        # TODO we can decide a better warm-up period,
        # since discarding 1000 customers sometimes leads to rejection of H0 :c
        warm_up = 500

        # theoretical waiting time:
        theoretical_wt = (
            arrival_rate * (service_time_std**2 + service_time_mean**2)
        ) / (2 * (1 - utilization))

        # R replications
        R = 40
        (
            passengers_passed,
            mean_waiting_time,
            std_waiting_time,
            waiting_times_collector,
        ) = run_multiple_simulations(
            arrival_rate=arrival_rate, num_replications=R, warm_up=warm_up
        )

        # hypothesis test
        t_value = 2.021
        validity = check_validity(
            mean_waiting_time,
            std_waiting_time,
            len(waiting_times_collector),
            theoretical_wt,
            t_value,
        )

        print("------SECURITY LANE SIMULATION------")
        print("------------Q2A  RESULTS------------")

        print(f"Mean passengers passed: {np.mean(passengers_passed):.0f}")
        print(
            f"Mean waiting time: {mean_waiting_time:.2f} minutes,\n",
            f"Standard deviation of waiting time: {std_waiting_time} minutes",
        )
        print(f"theoretical steady-state solution: {theoretical_wt}")
        print(f"Not reject H0: {validity}")

        """  plot_waiting_times_cumavg(
            waiting_times_collector, 20, warm_up, save_path="img/cumavg_2A.png"
        ) """

    if WARM_UP_SWEEP:
        utilization = 0.85
        service_time_mean = 1
        service_time_std = 0.25
        arrival_rate = utilization / service_time_mean

        warm_ups = range(0, 1001, 10)
        R = 1000

        stats_collector = []
        for warm_up in warm_ups:
            print(f"Running warm-up period: {warm_up}")
            _, mean, std, _ = run_multiple_simulations(
                arrival_rate=arrival_rate, num_replications=R, warm_up=warm_up
            )

            stats_collector.append((warm_up, mean, std))

        stats_collector = np.array(stats_collector)
        np.save("assignment_2/data/warm_up_sweep.npy", stats_collector)

    if PLOT_WARM_UP_SWEEP:

        stats_collector = np.load("assignment_2/data/warm_up_sweep.npy")
        warm_ups = stats_collector[:, 0]
        means = stats_collector[:, 1]
        stds = stats_collector[:, 2]

        utilization = 0.85
        service_time_mean = 1
        service_time_std = 0.25
        arrival_rate = utilization / service_time_mean

        theoretical_wt = (
            arrival_rate * (service_time_std**2 + service_time_mean**2)
        ) / (2 * (1 - utilization))

        fig, axs = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
        axs.plot(warm_ups, means, label=r"$\bar{X}$", lw=1.5)
        axs.fill_between(
            warm_ups,
            means - stds,
            means + stds,
            alpha=0.2,
            label=r"$s$",
        )
        axs.hlines(
            theoretical_wt,
            xmin=warm_ups[0],
            xmax=warm_ups[-1],
            colors="red",
            linestyles="--",
            label=r"$\mu$",
            lw=1,
        )

        # true value
        axs.set_xlabel("Warm-up period (customers)")
        axs.set_ylabel("Mean waiting time (minutes)")
        axs.grid(alpha=0.5)
        axs.set_title("Effect of warm-up period on waiting time estimates (R = 1000)")
        axs.legend()
        plt.tight_layout()
        plt.savefig("assignment_2/img/warm_up_sweep.png", dpi=300)
        plt.show()

    if Q2B:
        arrival_rate = get_arrival_rate(df, "September")
        R = 40

        # CURRENT OPERATIONS
        # R replications

        R = 40
        (
            passengers_passed,
            mean_waiting_time,
            std_waiting_time,
            waiting_times_collector,
        ) = run_multiple_simulations(arrival_rate=arrival_rate, num_replications=R)

        print("------SECURITY LANE SIMULATION------")
        print("------------Q2B  RESULTS------------")
        print("---------CURRENT OPERATIONS---------")

        print(f"Mean passengers passed: {np.mean(passengers_passed):.0f}")
        print(
            f"Mean waiting time: {mean_waiting_time:.2f} minutes,\n",
            f"Standard deviation of waiting time: {std_waiting_time} minutes",
        )

        # ADD MORE SERVERS
        print("-------------2  SERVERS-------------")

        (
            passengers_passed,
            mean_waiting_time,
            std_waiting_time,
            waiting_times_collector,
        ) = run_multiple_simulations(
            arrival_rate=arrival_rate, num_replications=R, num_servers=2
        )

        print(f"Mean passengers passed: {np.mean(passengers_passed):.0f}")
        print(
            f"Mean waiting time: {mean_waiting_time:.2f} minutes,\n",
            f"Standard deviation of waiting time: {std_waiting_time} minutes",
        )

        #  REDUCE VARIANCE
        print("--------REDUCED  VARIABILITY--------")

        (
            passengers_passed,
            mean_waiting_time,
            std_waiting_time,
            waiting_times_collector,
        ) = run_multiple_simulations(
            arrival_rate=arrival_rate, st_std=2, num_replications=R
        )

        print(f"Mean passengers passed: {np.mean(passengers_passed):.0f}")
        print(
            f"Mean waiting time: {mean_waiting_time:.2f} minutes,\n",
            f"Standard deviation of waiting time: {std_waiting_time} minutes",
        )
