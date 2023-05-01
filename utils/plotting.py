import wandb
import pandas as pd
import seaborn as sns
from config import configs
import matplotlib.pyplot as plt
ENTITY = "Scads"
if configs.get("usecase") == 2:
    ROUNDS = [1,2,4,8]
else:
    ROUNDS = [1,2,5,10]


def get_group_stats(project:str,groups:list,version:str,metric_names:list,mode:str):
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
        runs = api.runs(f"{ENTITY}/{project}",filters={"group": group})
        metrics_dicts = []
        for run in runs:
            if run.config.get("version") == version:
                metrics_dicts.append(get_metrics_from_run(run, metric_names,group,mode))
        metrics  = put_metrics_together(metrics_dicts)
        project_metrics[group] = metrics
    return project_metrics



def get_stats_for_usecase(groups,version = None,mode="balanced",rounds=None):
    """
    Get the mean of the metrics for each group of all projecets for that usecase
    :param mode: mode of the project can be "unweighted" or "system" or "balanced"
    :param groups: groups to get the mean of the metrics for
    :param version: version of the runs
    :return: list of dictionaries with the mean of the metrics for each group for each round
    """
    usecase = configs.get("usecase")
    project_prefix = "usecase"
    metrics_prefix = "model"
    data_path = configs.get("data_path").split("/")[-1].split(".")[0]
    if rounds is None:
        rounds  = ROUNDS
    round_metrics = []
    if not version:
        version = configs.get("version")
    if mode == "system":
        metrics_prefix ="system"
        version = None
        metrics_names = ["memory_client","memory_server","round_time","client_time"]
    else:
        metrics_names = [element.name for element in configs.get("metrics")]
        metrics_names.append("loss")
    if mode == "unweighted":
        project_prefix = "unweightedusecase"
        metrics_names.extend([name +"_global" for name in metrics_names])
    for round in rounds:
        project = f"{project_prefix}_{str(usecase)}_benchmark_rounds_{round}_{data_path}_{metrics_prefix}_metrics"
        round_metrics.append(get_group_stats(project,groups,version=version,metric_names=metrics_names,mode=mode))
    return round_metrics


def transform_scenario_metrics_to_df(metrics:dict,metric_name:str,round_num):
    """
    Transform the metrics for all roundconfiguration to a pandas dataframe for plotting
    :param metrics: metrics for all roundconfiguration
    :param metric_name: name of metric
    :round_num: number of rounds used for multiplying time metrics
    :return: df with the metrics for one roundconfiguration
    """
    dfs= []
    for groupname,metrics in metrics.items():
        for i in range(len(metrics[metric_name])):
            dfs.append(transform_to_df(metrics,metric_name,groupname.split("_")[0],groupname.split("_")[1],round_num, round_num))
    df = pd.concat(dfs)
    return df


def transform_to_df(metrics:dict,metric_name,framework,group,round_configuration,round_num=1):
    """
    Transform the metrics for one roundconfiguration and one metric to a pandas dataframe for plotting
    :param metrics: metrics for one roundconfiguration
    :param metric_name: name of metric
    :param framework: framework of the run
    :param group: group of the run
    :param round_configuration: roundconfiguration of the run
    :param round_num: number of rounds used for multiplying time metrics
    :return: df with the metrics for one roundconfiguration and one metric
    """
    rows = []
    for i in range(len(metrics[metric_name])):
        row = {
            "framework": framework,
            "group": group,
            "round configuration": round_configuration
        }
        metric_value = metrics[metric_name][i] if "time" not in metric_name else metrics[metric_name][i] * round_num
        row["metric"] = metric_value
        rows.append(row)
    df = pd.DataFrame(data=rows, columns=["framework", "group", "metric", "round configuration"])
    return df



def create_dfs_for_fl_metric(rounds_metrics,metric_name:str):
    """
    Creates a dataframe for each roundconfiguration for one metric and appends them together
    :param rounds_metrics: metrics for each roundconfiguration
    :param metric_name: name of metric
    :return: df with all values for one metric
    """
    dfs = []
    for index, metric_for_number_of_rounds in enumerate(rounds_metrics):
        round_num = ROUNDS[index]
        df = transform_scenario_metrics_to_df(metric_for_number_of_rounds, metric_name, round_num)
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def seaborn_plot (x,metric_name,hue,data,palette,title,dodge=True,configuration_name="Number of Rounds"):
    """
    Plots a seaborn boxplot to easy exchange plottype
    :param x: x axis data
    :param metric_name: metric nane
    :param hue: hue data
    :param data: data to plot
    :param palette: color palette
    :param title: title of the plot
    :param dodge: dodge parameter
    :return:
    """
    ax = sns.boxplot(x=x, y="metric", hue=hue,
                data=data, palette=palette, dodge=dodge)
    ax.set_title(title)
    # set y axis title to metric name
    plt.ylabel(metric_name)
    # set x axis title to group name
    plt.xlabel(configuration_name)
    if configuration_name == "Percentage of chosen class":
        if configs.get("usecase") == 1 or configs.get("usecase") == 2:
            start_value = 50
        else:
            start_value = 20
        x_ticks = [(int(float(x)) *5 + start_value) for x in data[x].unique() if x != "central"]
        x_ticks.append("central")
        ax.set_xticklabels(x_ticks)

    plt.show()

def plot_swarmplots(df,metric_name:str,configuration_name:str):
    """
    Plots swarmplots for the given df which contains all data for onem metric
    :param df: df with all data for one metric
    :metric_name: name of metric to plot
    """
    for index,round_num in enumerate(ROUNDS):
        round_df = df[df["round configuration"] == "central"]
        round_df = pd.concat([round_df,df[df["round configuration"] == round_num]],ignore_index=True)
        seaborn_plot("group", metric_name, "framework", round_df, "Set2",
                     f"Round configuration {round_num}",configuration_name=configuration_name)
        plt.show()
    seaborn_plot("group", metric_name, "framework", df, "Set2", f"Round configuration summarized",
                 configuration_name=configuration_name)
    seaborn_plot("round configuration", metric_name, "framework", df, "Set2", f"Group summarized")
    for group in df["group"].unique():
        if group == "central":
            continue
        group_df = df[df["group"] == "central"]
        group_df = pd.concat([group_df,df[df["group"] == group]],ignore_index=True)
        seaborn_plot("round configuration", metric_name, "framework", group_df, "Set2", f"Group {group}")



def get_central_metrics(mode:str,metric_names:list):
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
        metric_names = ["training_time","memory_central"]
        version = None
    else:
        project = "central_model_metrics"
        version = "essential_seeds_42"
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{project}",filters={"group": group})
    metrics_dicts = []
    for run in runs:
        if run.name != "no_crossfold" and run.config.get("version") == version:
            metrics_dicts.append(get_metrics_from_run(run,metric_names,group,mode))
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
    for metric in metric_names:
        if metric in ["memory_client","memory_server","memory_central"]:
            # if the metric is a memory metric, get the sum, mean and max of the metric
            memory_metrics = get_memory_metrics(group, history, metric)
            metrics.update(memory_metrics)
        elif metric in ["round_time","client_time","training_time"]:
            # if the metric is a time metric, get the mean of the metric
            time = history.get(metric)
            if time is not None:
                metrics[metric] = time.mean()
                if metric == "client_time":
                    # if the metric is client_time, get the first_round_time
                    metrics["first_round_time"] = time.iloc[0]
    if "client_time" in metrics.keys() and "round_time" in metrics.keys():
        # if both client_time and round_time are in the metric_names, get the time_diff
        metrics["time_diff"] = metrics["round_time"] - metrics["client_time"]
    if "total_memory_client" in metrics.keys() and "total_memory_server" in metrics.keys():
        # if both memory_client and memory_server are in metric_names calculate the sum
        # of both metrics summed up
        metrics["total_memory_fl"] = metrics["total_memory_client"] + metrics["total_memory_server"]
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
        memory_metrics[f"mean_{metric}"] = memory.mean() * number_of_clients
        memory_metrics[f"max_{metric}"] = memory.max() * number_of_clients
    return memory_metrics
def get_metrics_from_run(run:wandb.run,metric_names:list,group,mode:str):
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
        return get_system_metrics(history,metric_names,group)
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





def put_metrics_together(metrics_dicts:list):
    """
    Put the metrics of all runs into one dictionary
    :param metrics_dicts: list of dictionaries with the metrics for each run
    :return: dict with the averaged metrics for all runs
    """
    unique_keys = set()
    for metrics_dict in metrics_dicts:
        unique_keys.update(metrics_dict.keys())
    metrics = {key: [metrics_dict.get(key) for metrics_dict in metrics_dicts if metrics_dict.get(key) is not None]
               for key in unique_keys}
    return metrics

