import numpy as np
import pandas as pd
import wandb
from config import configs
from plotting import ENTITY, ROUNDS


def get_loss_stats(groups: list, version: str, mode: str):
    """
    Get the loss stats for the given groups and version
    :param groups: groups to get the stats for
    :param version: version to get the stats for
    :param mode: mode to get the stats for
    :return:
    """
    data_path = configs.get("data_path").split("/")[-1].split(".")[0]
    usecase = configs.get("usecase")
    project_prefix = "usecase"
    if mode == "unweighted":
        project_prefix = "unweightedusecase"
    project = f"{project_prefix}_{str(usecase)}_benchmark_rounds_{10}_{data_path}_model_metrics"
    if mode == "central":
        project = "central_model_metrics"
    api = wandb.Api()
    project_metrics = {}
    for group in groups:
        runs = api.runs(f"{ENTITY}/{project}", filters={"group": group})
        losses = []
        for run in runs:
            if (
                run.config.get("version") == version
                or run.config.get("version") == "unbalanced_with_global_evaluation_1804"
                and run.name != "no_crossfold"
            ):
                history = run.history()
                if mode == "unweighted":
                    loss = history.get("loss_global")
                elif mode == "central" and run.name != "no_crossfold":
                    loss = history.get("val_loss")
                else:
                    loss = history.get("loss")
                if run.state == "crashed":
                    continue
                else:
                    losses.append(loss.dropna().tolist())
        means = np.mean(np.array(losses), axis=0).tolist()
        project_metrics[group] = means
    return project_metrics


def create_loss_df(metrics_dict):
    """

    :param metrics_dict:
    :return:
    """
    total_rows = []
    for key, value in metrics_dict.items():
        if key.startswith("usecase"):
            group = "central"
            framework = "Centralized"
        else:
            group = key.split("_")[1]
            framework = key.split("_")[0].upper()
        rows = []
        for i in range(len(value)):
            rows.append(
                {
                    "round": i + 1,
                    "loss": value[i],
                    "group": group,
                    "framework": framework,
                }
            )
        total_rows.extend(rows)
    df = pd.DataFrame(total_rows)
    return df


def get_group_stats(
    project: str, groups: list, version: str, metric_names: list, mode: str
):
    """
    Get the mean of the metrics for each group of a project
    :param project: name of the project
    :param groups: names of the groups
    :param version: version of the runs
    :param metric_names: metrics to get the mean of
    :param mode: mode of the project can be "unweighted" or "system" or "balanced"
    :return: dictionary with the mean of the metrics for each group
    """
    api = wandb.Api()
    project_metrics = {}
    for group in groups:
        runs = api.runs(f"{ENTITY}/{project}", filters={"group": group})
        metrics_dicts = []
        for run in runs:
            if (
                run.config.get("version") == version or version is None
            ) and run.state != "failed":
                metrics_dicts.append(
                    get_metrics_from_run(run, metric_names, group, mode)
                )
        metrics = put_metrics_together(metrics_dicts)
        project_metrics[group] = metrics
    return project_metrics


def group_scenarios(scenarios: list, group_factor):
    """
    Group a list of dataframes by a column and return a df with the means over that column
    """
    df = pd.concat(scenarios)
    df = df.groupby([df.index, "framework", group_factor], as_index=False).agg(
        {"metric": "mean"}
    )
    return df


def get_stats_for_usecase(groups, version=None, mode="balanced", rounds=None):
    """
    Get the mean of the metrics for each group of all projecets for that usecase
    :param mode: mode of the project can be "unweighted" or "system" or "balanced"
    :param groups: groups to get the mean of the metrics for
    :param version: version of the runs
    :param rounds: rounds to get the metrics for
    :return: list of dictionaries with the mean of the metrics for each group for each round
    """
    usecase = configs.get("usecase")
    project_prefix = "usecase"
    metrics_prefix = "model"
    data_path = configs.get("data_path").split("/")[-1].split(".")[0]
    if rounds is None:
        rounds = ROUNDS
    round_metrics = []
    if not version:
        version = configs.get("version")
    if mode == "system":
        metrics_prefix = "system"
        version = "version_1005"
        metrics_names = [
            "memory_client",
            "memory_server",
            "round_time",
            "client_time",
            "sent",
            "received",
        ]
    else:
        metrics_names = [element.name for element in configs.get("metrics")]
        metrics_names.append("loss")
    if mode == "unweighted":
        project_prefix = "unweightedusecase"
        metrics_names.extend([name + "_global" for name in metrics_names])
    for round in rounds:
        project = f"{project_prefix}_{str(usecase)}_benchmark_rounds_{round}_{data_path}_{metrics_prefix}_metrics"
        round_metrics.append(
            get_group_stats(
                project, groups, version=version, metric_names=metrics_names, mode=mode
            )
        )
    return round_metrics


def transform_scenario_metrics_to_df(metrics: dict, metric_name: str, round_num):
    """
    Transform the metrics for all roundconfiguration to a pandas dataframe for plotting
    :param metrics: metrics for all roundconfiguration
    :param metric_name: name of metric
    :round_num: number of rounds used for multiplying time metrics
    :return: df with the metrics for one roundconfiguration
    """
    dfs = []
    for groupname, metrics in metrics.items():
        dfs.append(
            transform_to_df(
                metrics,
                metric_name,
                groupname.split("_")[0],
                groupname.split("_")[1],
                round_num,
            )
        )
    df = pd.concat(dfs)
    return df


def transform_to_df(metrics: dict, metric_name, framework, group, round_configuration):
    """
    Transform the metrics for one roundconfiguration and one metric to a pandas dataframe for plotting
    :param metrics: metrics for one roundconfiguration
    :param metric_name: name of metric
    :param framework: framework of the run
    :param group: group of the run
    :param round_configuration: roundconfiguration of the run
    :return: df with the metrics for one roundconfiguration and one metric
    """
    rows = []
    if framework in ["tff", "flwr"]:
        framework = framework.upper()
    if framework == "central":
        framework = "Centralized"
    for i in range(len(metrics[metric_name])):
        row = {
            "framework": framework,
            "group": group,
            "round configuration": round_configuration,
        }
        metric_value = metrics[metric_name][i]
        row["metric"] = metric_value
        rows.append(row)
    df = pd.DataFrame(
        data=rows, columns=["framework", "group", "metric", "round configuration"]
    )
    return df


def create_dfs_for_fl_metric(rounds_metrics, metric_name: str):
    """
    Creates a dataframe for each roundconfiguration for one metric and appends them together
    :param rounds_metrics: metrics for each roundconfiguration
    :param metric_name: name of metric
    :return: df with all values for one metric
    """
    dfs = []
    for index, metric_for_number_of_rounds in enumerate(rounds_metrics):
        round_num = ROUNDS[index]
        df = transform_scenario_metrics_to_df(
            metric_for_number_of_rounds, metric_name, round_num
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def recalculate_round_times_for_number_of_rounds(df):
    """
    Recalculates the round times for the given df so it is comparable for different number of rounds
    and the central model
    :param df: df with all data for round time
    :return: df with recalculated round times
    """
    df["metric"] = df["metric"] * df["round configuration"].astype(int)
    return df


def get_central_metrics(mode: str, metric_names: list):
    """
    Get the metrics for the central model
    :param mode: metric mode of the project can be "system" or "model"
    :param metric_names: names of the metrics to get
    :return: dictionary with the metrics for the central model
    """
    usecase = configs.get("usecase")
    group = f"usecase_{usecase}"
    if mode == "system":
        project = "benchmark-central_system_metrics"
        metric_names = ["training_time", "memory_central"]
        version = None
    else:
        project = "central_model_metrics"
        version = "essential_seeds_42"
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{project}", filters={"group": group})
    metrics_dicts = []
    for run in runs:
        if (
            run.name != "no_crossfold"
            and run.config.get("version") == version
            and run.state != "failed"
        ):
            metrics_dicts.append(get_metrics_from_run(run, metric_names, group, mode))
    metrics = put_metrics_together(metrics_dicts)
    return metrics


def get_system_metrics(history, metric_names, group):
    """
    Get the system metrics from the history of a run
    :param history: history of a run
    :param metric_names: metric names to get
    :param group: group of the run
    :return: dictionary with the metrics
    """
    metrics = {}
    for metric in set(metric_names).intersection(history.keys()):
        if metric in ["memory_client", "memory_server", "memory_central"]:
            # if the metric is a memory metric, get the sum, mean and max of the metric
            memory_metrics = get_memory_metrics(group, history, metric)
            metrics.update(memory_metrics)
        elif metric in ["round_time", "client_time", "training_time"]:
            # if the metric is a time metric, get the mean of the metric
            time = history.get(metric)
            if time is not None:
                metrics[metric] = time.mean()

                if metric == "client_time":
                    # if the metric is client_time, get the first_round_time
                    metrics["first_round_time"] = time.iloc[0]
                if metric == "client_time" or metric == "round_time":
                    metrics["total_" + metric] = time.sum()
        elif metric in ["sent", "received"]:
            metric_value = float(history.get(metric)[0])
            if metric_value is not None:
                number_of_clients = int(group.split("_")[-1])
                metrics[metric] = metric_value * number_of_clients
    if "client_time" in metrics.keys() and "round_time" in metrics.keys():
        # if both client_time and round_time are in the metric_names, get the time_diff
        metrics["time_diff"] = metrics["round_time"] - metrics["client_time"]
        metrics["time_diff_percentage"] = metrics["time_diff"] / metrics["client_time"]
    if (
        "total_memory_client" in metrics.keys()
        and "total_memory_server" in metrics.keys()
    ):
        # if both memory_client and memory_server are in metric_names calculate the sum
        # of both metrics summed up
        metrics["total_memory_fl"] = (
            metrics["total_memory_client"] + metrics["total_memory_server"]
        )
    return metrics


def get_memory_metrics(group, history, metric):
    """
    Get the sum, mean and max of a memory metric
    :param group: group of the run
    :param history: history of the run
    :param metric: name of the metric
    :return: dictionary with the sum, mean and max of the metric
    """
    memory_metrics = {}
    memory = history.get(metric)
    if memory is not None:
        if metric == "memory_client":
            number_of_clients = int(group.split("_")[-1])
        else:
            number_of_clients = 1
        memory_metrics[f"total_{metric}"] = memory.sum() * number_of_clients
        memory_metrics[f"total_per_client_{metric}"] = memory.sum()
        memory_metrics[f"mean_per_client_{metric}"] = memory.mean()
        memory_metrics[f"max_per_client_{metric}"] = memory.max()
        memory_metrics[f"mean_{metric}"] = memory.mean() * number_of_clients
        memory_metrics[f"max_{metric}"] = memory.max() * number_of_clients
    return memory_metrics


def get_metrics_from_run(run: wandb.run, metric_names: list, group, mode: str):
    """
    Get the metrics from a run
    :param run: run to get the metrics from
    :param metric_names: names of the metrics to get
    :param group: group of the run
    :param group: mode of the run
    :param mode: mode of the run
    :return: dictionary with the metrics for the run
    """
    history = None
    metrics = {}
    if mode == "system":
        history = run.history()
        return get_system_metrics(history, metric_names, group)
    for metric in metric_names:
        if metric not in run.summary.keys():
            # If the run summary does not contain the metric, get it from the history
            if history is None:
                history = run.history()
            if history.get(metric) is not None:
                metrics[metric] = history.get(metric).iloc[-1]
        else:
            # If the run summary contains the metric, get it from the summary
            metrics[metric] = run.summary.get(metric)
    return metrics


def put_metrics_together(metrics_dicts: list):
    """
    Put the metrics of all runs into one dictionary
    :param metrics_dicts: list of dictionaries with the metrics for each run
    :return: dict with the averaged metrics for all runs
    """
    unique_keys = set()
    for metrics_dict in metrics_dicts:
        unique_keys.update(metrics_dict.keys())
    metrics = {
        key: [
            metrics_dict.get(key)
            for metrics_dict in metrics_dicts
            if metrics_dict.get(key) is not None
        ]
        for key in unique_keys
    }
    return metrics