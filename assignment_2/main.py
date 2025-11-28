import itertools
import os

os.makedirs("assignment_2/img", exist_ok=True)
os.makedirs("assignment_2/data", exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy
from scipy import stats
from tqdm import tqdm

# Global visual style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (10, 6),
        "lines.linewidth": 1,
    }
)


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

        # ensure non-negative service time
        self.service_times = stats.truncnorm.rvs(
            -self.service_time_mean / self.service_time_standard_dev,
            np.inf,
            loc=self.service_time_mean,
            scale=self.service_time_standard_dev,
            size=self.n_customers,
        )

        self.current_index = 0

    def service_passenger(self):
        service_time = self.service_times[self.current_index]
        self.current_index += 1

        yield self.env.timeout(service_time)


def passenger(
    env: simpy.Environment,
    name: str,
    queue_lengths,
    security_lane: SecurityLane,
    waiting_times: list,
    verbose: bool = False,
):
    """A single passsenger passing through the security lane-"""

    # set timer for arrival

    with security_lane.server.request() as request:
        queue_len = len(security_lane.server.queue)
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
        queue_lengths.append((env.now, queue_len))
        if verbose:
            print(f"{name} waited for {wait_queue_time:.2f} minutes")


def setup(
    env: simpy.Environment,
    num_servers: int,
    arrival_rate: float,
    waiting_times: list,
    queue_lengths: list,
    verbose: bool = False,
    st_std: float = 0.25,
    st_mean: float = 1.0,
    passengers_passed: int = 3000,
    init_passengers: int = 0,
):
    """Setting up the security lane simulation"""

    # create security lane
    security_lane = SecurityLane(
        env, num_servers, st_std=st_std, st_mean=st_mean, n_customers=passengers_passed
    )

    # add passengers that are already in the queue
    passenger_count = itertools.count()
    for _ in range(init_passengers):
        env.process(
            passenger(
                env,
                f"Passenger {next(passenger_count)}",
                queue_lengths,
                security_lane,
                waiting_times,
                verbose=verbose,
            )
        )

    # add more passengers while running
    passed = 0
    while passed < (passengers_passed - init_passengers):
        yield env.timeout(np.random.exponential(1 / arrival_rate))
        passenger_id = next(passenger_count)
        env.process(
            passenger(
                env,
                f"Passenger {passenger_id}",
                queue_lengths,
                security_lane,
                waiting_times,
                verbose=verbose,
            )
        )
        passed += 1


def run_simulation(
    arrival_rate: float,
    st_std: float = 0.25,
    st_mean: float = 1.0,
    verbose: bool = False,
    passengers_passed: int = 3000,
    num_servers: int = 1,
    init_passengers: int = 0,
    stop_time: int = 0,
):
    """Runs a single simulation"""
    waiting_times = []
    queue_lengths = []
    env = simpy.Environment()
    env.process(
        setup(
            env,
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            queue_lengths=queue_lengths,
            waiting_times=waiting_times,
            verbose=verbose,
            passengers_passed=passengers_passed,
            st_std=st_std,
            st_mean=st_mean,
            init_passengers=init_passengers,
        )
    )
    if stop_time:
        env.run(until=stop_time)
    else:
        env.run()
    return waiting_times, queue_lengths


def run_multiple_simulations(
    arrival_rate: float,
    st_mean: float = 1.0,
    st_std: float = 0.25,
    verbose: bool = False,
    passengers_passed: int = 3000,
    num_servers: int = 1,
    num_replications=40,
    warm_up: int = 0,
    init_passengers: list = 0,
    stop_time: int = 0,
):
    waiting_times_collector = []
    queue_length_collector = []

    for i, _ in tqdm(enumerate(range(num_replications))):
        wt, ql = run_simulation(
            arrival_rate,
            st_mean=st_mean,
            st_std=st_std,
            passengers_passed=passengers_passed,
            verbose=verbose,
            num_servers=num_servers,
            init_passengers=(
                init_passengers[i]
                if isinstance(init_passengers, list)
                else init_passengers
            ),
            stop_time=stop_time,
        )
        waiting_times_collector.append(wt)
        queue_length_collector.append(ql)

    # waiting_times_collector = np.array(waiting_times_collector, dtype=object)
    # queue_length_collector = np.array(queue_length_collector, dtype=object)

    queue_length_collector = [np.array(ql, dtype=int) for ql in queue_length_collector]
    waiting_times_collector = [
        np.array(wt, dtype=float) for wt in waiting_times_collector
    ]

    passengers_passed = [len(w) for w in waiting_times_collector]

    # apply warm-up
    trimmed = [ws[warm_up:] for ws in waiting_times_collector]
    mean_waiting_time = np.mean([np.mean(ws) for ws in trimmed])
    std_waiting_time = np.std([np.mean(ws) for ws in trimmed], ddof=1)

    return (
        passengers_passed,
        mean_waiting_time,
        std_waiting_time,
        waiting_times_collector,
        queue_length_collector,
    )


def check_validity(
    average: float, std: float, length: int, theoretical_value: float, t_value: float
) -> bool:

    t_stat = (average - theoretical_value) / (std / np.sqrt(length))
    return t_stat < t_value


def std_sweep(
    arrival_rate: float,
    std_values: float,
    R: int = 40,
    passengers_passed: int = 3000,
):
    means = []
    stds = []

    for st in std_values:
        print(f"Running sweep for σ = {st}")
        _, mean, sdev, _, _ = run_multiple_simulations(
            arrival_rate=arrival_rate,
            st_std=st,
            num_replications=R,
            passengers_passed=passengers_passed,
        )
        means.append(mean)
        stds.append(sdev)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(std_values, means, marker="o", label="Mean waiting time")
    plt.fill_between(
        std_values,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.2,
        label="±1 std dev",
    )
    plt.xlabel("Service time standard deviation σ")
    plt.ylabel("Waiting time (minutes)")
    plt.title("Effect of σ on Waiting Times (Part 2B)")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout(pad=1.2)
    plt.savefig("img/std_sweep.png", bbox_inches="tight")
    plt.show()


def plot_average_waiting_times(per_sim_means, scenario_name="Scenario"):
    plt.figure()

    plt.hist(per_sim_means, bins=15, alpha=0.6, density=True, label="Simulation means")

    mu = np.mean(per_sim_means)
    sigma = np.std(per_sim_means, ddof=1)
    x = np.linspace(min(per_sim_means), max(per_sim_means), 200)
    pdf = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, label=f"Normal fit (μ={mu:.2f}, σ={sigma:.2f})")

    plt.xlabel("Average waiting time per simulation (min)")
    plt.ylabel("Density")
    plt.title(f"Distribution — {scenario_name}")
    plt.legend()
    plt.tight_layout(pad=1.2)
    plt.show()
    plt.close()


def plot_queue_length_mean(queue_lengths_collector, scenario_name="Scenario"):
    max_time = max(ql[-1][0] for ql in queue_lengths_collector)
    common_times = np.linspace(0, max_time, 500)

    interpolated = []
    for ql in queue_lengths_collector:
        times = np.array([t for t, _ in ql])
        lengths = np.array([q for _, q in ql])
        interpolated.append(np.interp(common_times, times, lengths))

    interpolated = np.array(interpolated)
    mean_curve = np.mean(interpolated, axis=0)
    sem_curve = np.std(interpolated, axis=0, ddof=1) / np.sqrt(len(interpolated))

    plt.figure()
    plt.plot(common_times, mean_curve, label="Mean queue length")
    plt.fill_between(
        common_times,
        mean_curve - 1.96 * sem_curve,
        mean_curve + 1.96 * sem_curve,
        alpha=0.25,
        label="95% CI",
    )

    plt.xlabel("Time (minutes)")
    plt.ylabel("Mean Queue Length")
    plt.title(f"Mean Queue Length — {scenario_name}")
    plt.legend()
    plt.tight_layout(pad=1.2)
    plt.show()
    plt.close()


def plot_queue_length_single(queue_lengths, scenario_name="Scenario 1"):
    times = [t for t, q in queue_lengths]
    lengths = [q for t, q in queue_lengths]

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(times, lengths, linewidth=1.6)

    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Queue Length", fontsize=12)
    plt.title(f"Queue Length Over Time — {scenario_name}", fontsize=14)
    plt.grid(alpha=0.35, linestyle="--")
    plt.tight_layout(pad=2.0)
    plt.show()
    plt.close()


def plot_queue_lengths_all(queue_lengths_collector, scenario_name="Scenario"):
    plt.figure(figsize=(12, 6), dpi=300)

    for ql in queue_lengths_collector:
        times = [t for t, q in ql]
        lengths = [q for t, q in ql]
        plt.plot(times, lengths, alpha=0.2, linewidth=1)

    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Queue Length", fontsize=12)
    plt.title(
        f"Queue Length Trajectories Across Replications — {scenario_name}", fontsize=14
    )
    plt.grid(alpha=0.35, linestyle="--")
    plt.tight_layout(pad=2.0)
    plt.show()
    plt.close()


def plot_waiting_times_cumavg(
    waiting_times_collector: list,
    reps_to_plot: int,
    warm_up: int = 0,
    save_path: str = "assignment_2/img/",
    show: bool = False,
):
    plt.figure()

    # Individual trajectories
    for r in range(reps_to_plot):
        cumavg = np.cumsum(waiting_times_collector[r]) / np.arange(
            1, len(waiting_times_collector[r]) + 1
        )
        plt.plot(cumavg, alpha=0.35, linewidth=1)

    # Ensemble statistic
    avg_per_customer = np.mean(waiting_times_collector, axis=0)
    cumavg_per_customer = np.cumsum(avg_per_customer) / np.arange(
        1, len(avg_per_customer) + 1
    )
    std_per_customer = np.std(waiting_times_collector, axis=0) / np.sqrt(
        len(waiting_times_collector)
    )
    cumavg_std = np.cumsum(std_per_customer) / np.arange(1, len(std_per_customer) + 1)

    plt.plot(
        cumavg_per_customer, color="black", label="ensemble-averaged cumulative mean"
    )
    plt.fill_between(
        np.arange(len(cumavg_per_customer)),
        cumavg_per_customer - cumavg_std,
        cumavg_per_customer + cumavg_std,
        alpha=0.25,
        color="gray",
        label="95% confidence band",
    )

    if warm_up:
        plt.axvline(x=warm_up, color="red", label="warm up", ls="--")

    plt.xlabel("Customer index")
    plt.ylabel("Cumulative average waiting time (min)")
    plt.legend()
    plt.tight_layout(pad=1.2)

    if save_path:
        plt.savefig(save_path + "miao.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_std_sweep2(
    arrival_rate: float,
    st_std_range: list[float],
    num_servers: int,
    n_steps: int = 1000,
    passed_passengers: int = 3000,
    num_replications: int = 40,
    show: bool = False,
    save_path: str = "assignment_2/img/",
):
    std_list = np.linspace(st_std_range[0], st_std_range[1], n_steps)
    mean_collector = np.zeros_like(std_list)
    ci_halfwidth_collector = np.zeros_like(std_list)

    for idx, st_std in enumerate(std_list):
        (
            _,
            mean_waiting_time,
            std_waiting_time,
            waiting_times_collector,
            _,
        ) = run_multiple_simulations(
            arrival_rate=arrival_rate,
            st_std=st_std,
            verbose=False,
            passengers_passed=passed_passengers,
            num_servers=num_servers,
            num_replications=num_replications,
            warm_up=0,
        )

        # Store mean
        mean_collector[idx] = mean_waiting_time

        # 95% confidence interval for mean across replications
        ci_halfwidth_collector[idx] = (
            1.96 * std_waiting_time / np.sqrt(num_replications)
        )

    lower = mean_collector - ci_halfwidth_collector
    upper = mean_collector + ci_halfwidth_collector

    # Plot mean with CI band
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(std_list, mean_collector, label="Mean waiting time")
    plt.fill_between(
        std_list, lower, upper, alpha=0.25, label="95% Confidence Interval"
    )

    plt.xlabel("Service time standard deviation σ")
    plt.ylabel("Average waiting time (minutes)")
    plt.title(f"Effect of σ on Waiting Times ({num_servers} Servers)")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout(pad=1.2)

    if save_path:
        plt.savefig(save_path + "sweep_with_CI.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot3D_std_mean_sweep(
    arrival_rate: float,
    std_range: list[float],
    mu_range: list[float],
    num_servers: int,
    passed_passengers: int = 3000,
    num_replications: int = 40,
    show: bool = False,
    save_path: str = "assignment_2/img/",
):

    mean_wt_matrix = np.zeros((len(std_range), len(mu_range)))
    for std_idx, std in enumerate(std_range):
        for mu_idx, mu in enumerate(mu_range):
            _, mean_waiting_time, _, _, _ = run_multiple_simulations(
                arrival_rate=arrival_rate,
                st_std=std,
                st_mean=mu,
                verbose=False,
                passengers_passed=passed_passengers,
                num_servers=num_servers,
                num_replications=num_replications,
                warm_up=0,
            )
            mean_wt_matrix[std_idx, mu_idx] = mean_waiting_time

    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(mu_range, std_range)
    surf = ax.plot_surface(
        Y,
        X,
        mean_wt_matrix,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.25,
        antialiased=True,
    )

    ax.set_xlabel("Service Time Std Dev (σ)", fontsize=11, labelpad=12)
    ax.set_ylabel("Service Time Mean (μ)", fontsize=11, labelpad=12)
    ax.set_zlabel("Mean Waiting Time (min)", fontsize=11, labelpad=10)
    ax.set_title(
        f"Mean Waiting Time Surface Plot ({num_servers} Servers)", fontsize=14, pad=15
    )

    fig.colorbar(surf, shrink=0.7, aspect=15, pad=0.1)
    plt.tight_layout(pad=2.5)

    if save_path:
        plt.savefig(
            save_path + "plot3D_std_mean_sweep.png", dpi=300, bbox_inches="tight"
        )

    if show:
        plt.show()
    plt.close()


def plot_warm_up_sweep(utilization, service_time_mean, service_time_std, arrival_rate):

    stats_collector = np.load("assignment_2/data/warm_up_sweep.npy")
    warm_ups = stats_collector[:, 0]
    means = stats_collector[:, 1]
    stds = stats_collector[:, 2]

    theoretical_wt = (
        arrival_rate * (service_time_std**2 + service_time_mean**2)
    ) / (2 * (1 - utilization))

    fig, axs = plt.subplots(1, 1, figsize=(8, 5), dpi=300, constrained_layout=True)
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
    plt.tight_layout(pad=1.2)
    plt.savefig("assignment_2/img/warm_up_sweep.png", dpi=300, bbox_inches="tight")
    plt.show()


def run_Q2A(df):
    utilization = 0.85
    service_time_mean = 1
    service_time_std = 0.25
    arrival_rate = utilization / service_time_mean
    warm_up = 500

    theoretical_wt = (
        arrival_rate * (service_time_std**2 + service_time_mean**2)
    ) / (2 * (1 - utilization))

    R = 40
    (
        passengers_passed,
        mean_waiting_time,
        std_waiting_time,
        waiting_times_collector,
        queue_lengths_collector,
    ) = run_multiple_simulations(
        arrival_rate=arrival_rate, num_replications=R, warm_up=warm_up
    )

    print("------SECURITY LANE SIMULATION------")
    print("------------Q2A  RESULTS------------")
    print(f"Mean passengers passed: {np.mean(passengers_passed):.0f}")
    print(
        f"Mean waiting time: {mean_waiting_time:.2f} minutes,\n"
        f"Standard deviation: {std_waiting_time:.2f} minutes"
    )
    print(f"Theoretical steady-state: {theoretical_wt:.3f}")
    print(
        "Not reject H0:",
        check_validity(
            mean_waiting_time, std_waiting_time, R, theoretical_wt, t_value=2.021
        ),
    )

    plot_nservers_cumavg(
        arrival_rate=arrival_rate,
        num_servers_list=[1],
        reps_to_plot=20,
        warm_up=warm_up,
        save_path="assignment_2/img/",
        show=False,
    )


def run_Q2B(df, plot_std_sweep=False):
    warm_up = 0
    arrival_rate = get_arrival_rate(df, "September")
    R = 40

    (
        passengers_passed,
        mean_waiting_time,
        std_waiting_time,
        waiting_times_collector,
        queue_lengths_collector,
    ) = run_multiple_simulations(arrival_rate=arrival_rate, num_replications=R)

    per_sim_means = np.array([np.mean(ws) for ws in waiting_times_collector])

    plot_average_waiting_times(per_sim_means, "Baseline")
    plot_queue_length_single(queue_lengths_collector[0])
    plot_queue_lengths_all(queue_lengths_collector, "Baseline")
    plot_queue_length_mean(queue_lengths_collector, "Baseline")

    print("------Q2B RESULTS: CURRENT OPS------")
    print(f"Mean passengers passed: {np.mean(passengers_passed):.0f}")
    print(f"Mean waiting: {mean_waiting_time:.2f} min")
    print(f"Std waiting:  {std_waiting_time:.2f} min")

    # Add servers
    run_multiple_simulations(arrival_rate, num_servers=2, passengers_passed=3500)

    plot_nservers_cumavg(
        num_servers_list=[1, 2, 3],
        arrival_rate=arrival_rate,
        passed_passengers=3000,
        save_path="assignment_2/img/",
        show=False,
        num_replications=R,
        colors=["blue", "orange", "green"],
    )

    plot_waiting_times_cumavg(
        waiting_times_collector,
        reps_to_plot=20,
        warm_up=0,
        save_path="assignment_2/img/",
        show=False,
    )

    # Standard deviation sweep with CI bands
    plot_std_sweep2(
        arrival_rate,
        [0.05, 1.0],
        num_servers=1,
        n_steps=100,
        num_replications=40,
        save_path="assignment_2/img/CI_sweep.png",
    )

    if plot_std_sweep:
        std_values = np.linspace(0.05, 1.0, 100)
        std_sweep(arrival_rate, std_values)


def run_3D_sweep(df):
    arrival_rate = get_arrival_rate(df, "September")
    num_servers = 3

    max_rho = 1.1
    max_mu = max_rho / arrival_rate * num_servers
    min_mu = 0.5 * max_mu
    max_std = 0.8 * max_mu
    min_std = 0.2 * min_mu

    mu_range = np.linspace(min_mu, max_mu, 10)
    std_range = np.linspace(min_std, max_std, 10)

    print(f"mu range: {mu_range}")
    print(f"std range: {std_range}")
    print(f"arrival rate: {arrival_rate}")

    plot3D_std_mean_sweep(
        arrival_rate,
        std_range,
        mu_range,
        num_servers,
        passed_passengers=3000,
        num_replications=40,
        save_path="assignment_2/img/",
        show=False,
    )


def run_2D_std_sweep(df):
    """Runs the 2D sweep of waiting time vs service-time std dev."""
    arrival_rate = get_arrival_rate(df, "September")
    servers_to_test = [4, 5]
    R = 40

    print("------ 2D STD SWEEP ------")
    print(f"Arrival rate: {arrival_rate}")

    for n in servers_to_test:
        print(f"Running STD sweep for {n} servers...")
        plot_std_sweep2(
            arrival_rate=arrival_rate,
            st_std_range=[0.01, 1],
            num_servers=n,
            n_steps=300,  # keep lightweight unless needed
            passed_passengers=3000,
            num_replications=R,
            save_path=f"assignment_2/img/sweep_{n}_servers.png",
            show=False,
        )

    print("STD sweep complete. Plots saved to /img/")


def plot_hourly_arrivals():
    arrival_rates = np.array(
        [
            0.5,
            0.4,
            0.3,
            0.2,
            0.4,
            0.8,
            1.2,
            1.8,
            2.5,
            3.1,
            2.0,
            1.7,
            1.1,
            1.9,
            2.1,
            2.8,
            2.2,
            3.3,
            2.5,
            2.4,
            2.0,
            2.1,
            0.7,
            0.3,
        ]
    )
    R = 40
    max_passengers = 1000
    queue_lengths_collector = [
        np.zeros((max_passengers, 2), dtype=int) for _ in range(R)
    ]
    ql_for_plot = []
    wt_for_plot = []
    for arrival_rate in tqdm(arrival_rates):
        passing_ql = [ql[-1, 1] for ql in queue_lengths_collector]

        (
            passengers_passed,
            mean_waiting_time,
            std_waiting_time,
            waiting_times_collector,
            queue_lengths_collector,
        ) = run_multiple_simulations(
            arrival_rate=arrival_rate,
            num_replications=R,
            passengers_passed=max_passengers,
            init_passengers=passing_ql,
            verbose=False,
            stop_time=480,
            num_servers=2,
        )
        mean_ql = np.mean([ql[-1, 1] for ql in queue_lengths_collector])
        ql_for_plot.append(mean_ql)
        wt_for_plot.append(mean_waiting_time)

    fig, axs = plt.subplots(1, 1, figsize=(8, 5), dpi=300, constrained_layout=True)
    axs2 = axs.twinx()
    axs3 = axs.twinx()
    hours = np.arange(0, 24)

    # plot queue length
    axs.plot(
        hours,
        ql_for_plot,
        color="darkred",
        marker="s",
        label="Mean queue length",
        lw=1.5,
        ms=3,
    )
    axs.set_xlabel("Hour of the day")
    axs.set_ylabel("Mean queue length [passengers]", color="darkred")
    axs.set_zorder(2)
    axs.patch.set_alpha(0)

    # waiting time
    axs2.plot(
        hours,
        wt_for_plot,
        color="darkgreen",
        marker="o",
        label="Mean waiting time",
        lw=1.5,
        ms=3,
    )
    axs2.set_ylabel("Mean waiting time [minutes]", color="darkgreen")
    axs2.set_zorder(2)
    axs2.patch.set_alpha(0)

    # arrival rate
    bars = axs3.bar(hours, arrival_rates, width=0.8, alpha=0.6, label=r"$\lambda$")
    for bar in bars:
        height = bar.get_height()
        axs3.text(
            bar.get_x() + bar.get_width() / 2,  # x-position
            height + 0.05,  # y-position (middle of bar)
            f"{height:.1f}",  # text
            ha="center",
            va="center",
            color="darkblue",
            fontsize=6,
            alpha=0.6,
        )
    axs3.set_zorder(0)
    axs3.patch.set_visible(False)
    # Move axis 3 further right so it doesn't overlap axis 2
    axs3.spines["right"].set_position(("axes", 1.15))

    # Hide axis 3 visually
    axs3.spines["right"].set_visible(False)
    axs3.yaxis.set_visible(False)

    axs.set_xticks(hours)
    axs.set_xticklabels(
        [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ]
    )
    axs.tick_params(axis="x", rotation=45, labelsize=8)

    plt.suptitle("Hourly Arrival Rate and Mean Waiting Time (2 Servers)")
    plt.tight_layout(pad=1.2)
    plt.savefig(
        "assignment_2/img/arrival_rate_vs_waiting_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_nservers_cumavg(
    arrival_rate: float,
    num_servers_list: list[int] = [1],
    st_std: float = 0.25,
    passed_passengers: int = 3000,
    num_replications: int = 40,
    reps_to_plot: int = 0,
    warm_up: int = 0,
    show: bool = False,
    save_path: str = "assignment_2/img/",
    colors=["black", "red", "purple"],
):
    for idx in range(len(num_servers_list)):
        _, mean_waiting_time, _, waiting_times_collector, _ = run_multiple_simulations(
            arrival_rate=arrival_rate,
            st_std=st_std,
            verbose=False,
            passengers_passed=passed_passengers,
            num_servers=num_servers_list[idx],
            num_replications=num_replications,
            warm_up=0,
        )

        for r in range(reps_to_plot):
            cumavg = np.cumsum(waiting_times_collector[r]) / np.arange(
                1, len(waiting_times_collector[r]) + 1
            )
            plt.plot(cumavg, alpha=0.5, lw=0.5)

        avg_per_customer = np.mean(waiting_times_collector, axis=0)
        cumavg_per_customer = np.cumsum(avg_per_customer) / np.arange(
            1, len(avg_per_customer) + 1
        )
        plt.plot(
            cumavg_per_customer,
            label=f"n of servers: {num_servers_list[idx]}",
            color=colors[idx],
        )
        print(
            f"for {num_servers_list[idx]} servers, waiting time of: {mean_waiting_time}"
        )

    if warm_up:
        plt.axvline(x=warm_up, color="red", label="warm up", ls="--")

    plt.xlabel("customer index")
    plt.ylabel("cumulative average waiting time")
    plt.title("ensemble-averaged cumulative mean")
    plt.grid(alpha=0.5)
    plt.tight_layout(pad=1.2)
    plt.legend()

    if show:
        plt.show()

    # save if path is there
    if save_path:
        plt.savefig(save_path + "image_cumvag.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_warm_up(path: str, arrival_rate: float):

    warm_ups = range(0, 1001, 100)
    R = 40

    stats_collector = []
    for warm_up in warm_ups:
        print(f"Running warm-up period: {warm_up}")
        _, mean, std, _, _ = run_multiple_simulations(
            arrival_rate=arrival_rate, num_replications=R, warm_up=warm_up
        )

        stats_collector.append((warm_up, mean, std))

    stats_collector = np.array(stats_collector)
    np.save(path, stats_collector)


def main():
    df = load_data("airport.csv")
    Q2A = False
    WARM_UP_SWEEP = False
    Q2B = False
    HOURLY_ARRIVALS = True
    PLOT_STD_SWEEP = False
    PLOT_3D_MEAN_STD_SWEEP = False

    if Q2A:
        run_Q2A(df)

    if Q2B:
        run_Q2B(df)

    if PLOT_STD_SWEEP:
        run_2D_std_sweep(df)

    if WARM_UP_SWEEP:
        path = "assignment_2/data/warm_up_sweep.npy"
        utilization = 0.85
        service_time_mean = 1
        service_time_std = 0.25
        arrival_rate = utilization / service_time_mean

        save_warm_up(path, arrival_rate)

        plot_warm_up_sweep(
            utilization, service_time_mean, service_time_std, arrival_rate
        )

    # Varying arrival rates over the day
    if HOURLY_ARRIVALS:
        plot_hourly_arrivals()

    # 3D plot of mean waiting time vs std and mean of service time
    if PLOT_3D_MEAN_STD_SWEEP:
        run_3D_sweep(df)


if __name__ == "__main__":
    main()
