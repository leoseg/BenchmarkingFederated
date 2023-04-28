import wandb
import pandas as pd
import seaborn as sns
from config import configs
import matplotlib.pyplot as plt
ENTITY = "Scads"



def get_group_stats(project:str,groups:list,version:str,metric_names:list):
    """
    Get the mean of the metrics for each group of a project
    :param project: name of the project
    :param groups: names of the groups
    :param version: version of the runs
    :param metric_names: metrics to get the mean of
    :return: dictionary with the mean of the metrics for each group
    """
    api = wandb.Api()
    project_metrics = {}
    for group in groups:
        runs = api.runs(f"{ENTITY}/{project}",filters={"group": group})
        metrics_dicts = []
        for run in runs:
            if run.config.get("version") == version:
                # If the run summary does not contain one of the metrics, get it from the history
                if any([metric_name not in run.summary.keys() for metric_name in metric_names]):
                    history = run.history()
                    metrics_dicts.append({key:history.get(key).iloc[-1] for key in metric_names})
                # If the run summary contains the metric, get it from the summary
                else:
                    metrics_dicts.append({key:run.summary.get(key) for key in metric_names})
        metrics  = {key: [metrics_dict.get(key) for metrics_dict in metrics_dicts if metrics_dict.get(key) is not None ] for key in metric_names}
        project_metrics[group] = metrics
    return project_metrics



def get_stats_for_usecase(groups,version = None,mode="balanced",as_df=False):
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
    rounds  = [1,2,5,10]
    round_metrics = []
    if not version:
        version = configs.get("version")
    if mode == "system":
        metrics_prefix ="system"
        metrics_names = ["memory_client","memory_server","round_time","client_time"]
    else:
        metrics_names = [element.name for element in configs.get("metrics")]
        metrics_names.append("loss")
    if mode == "unweighted":
        project_prefix = "unweightedusecase"
        metrics_names.extend([name +"_global" for name in metrics_names])
    if usecase == 2:
        rounds = [1,2,4,8]
    for round in rounds:
        project = f"{project_prefix}_{str(usecase)}_benchmark_rounds_{round}_{data_path}_{metrics_prefix}_metrics"
        round_metrics.append(get_group_stats(project,groups,version=version,metric_names=metrics_names))
    return round_metrics


def transform_to_df(metrics_for_number_of_rounds:dict,metric_name:str,central_data=None):
    """
    Transform the metrics for one roundconfiguration to a pandas dataframe for plotting
    :param round_metrics:
    :return: df with the metrics for one roundconfiguration
    """
    rows = []
    for groupname,metrics in metrics_for_number_of_rounds.items():
        for i in range(len(metrics[metric_name])):
            row = {
                "framework":groupname.split("_")[0],
                "group":groupname.split("_")[1],
                metric_name:metrics[metric_name][i]
            }
            rows.append(row)
    if central_data:
        for i in range(len(central_data[metric_name])):
            row = {
                "framework":"central",
                "group":"central",
                metric_name:central_data[metric_name][i]
            }
            rows.append(row)
    df = pd.DataFrame(data = rows,columns=["framework", "group", metric_name])
    return df

def plot_swarmplots(rounds_metrics,metric_name:str,central_data=None):
    """
    Plots swarmplots for the metrics of all roundconfigurations
    :param df:
    :return:
    """
    for metric_for_number_of_rounds in rounds_metrics:
        df = transform_to_df(metric_for_number_of_rounds,metric_name,central_data)
        sns.boxplot(x="group", y=metric_name, hue="framework",
                    data=df, palette="Set2", dodge=True)
        plt.show()


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
    else:
        project = "central_model_metrics"
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{project}",filters={"group": group})
    metrics_dicts = []
    for run in runs:
        if run.name != "no_crossfold" and run.config.get("version") == "essential_seeds_42":
            # If the run summary does not contain one of the metrics, get it from the history
            if any([metric_name not in run.summary.keys() for metric_name in metric_names]):
                history = run.history()
                metrics_dicts.append({key: history.get(key).iloc[-1] for key in metric_names})
            # If the run summary contains the metric, get it from the summary
            else:
                metrics_dicts.append({key: run.summary.get(key) for key in metric_names})
    metrics = {key: [metrics_dict.get(key) for metrics_dict in metrics_dicts if metrics_dict.get(key) is not None]
               for key in metric_names}
    return metrics