from datapostprocessing.plotting import plot_swarmplots
from datapostprocessing.data_uploading_utils import (
    transform_to_df,
    create_dfs_for_fl_metric,
)
from config import configs
from datapostprocessing.db_utils import MongoDBHandler
import pandas as pd

"""
Plotting script for which plots all heatmaps and barplots for one usecase
"""

mongodb = MongoDBHandler()
for mode in ["system"]:
    plot_path = configs.get("plot_path") + mode + "/"
    unweighted = False
    metrics_tuples = []
    match mode:
        case "unweighted":
            # if configs.get("usecase") == 1 or configs.get("usecase") == 2:
            #     metric1 = "binary_accuracy_global"
            # else:
            #     metric1 = "sparse_categorical_accuracy_global"
            metric2 = "auc_global"
            # metric1_name = "Accuracy"
            metric2_name = "AUC"
            unweighted = True
            # metrics_tuples.append((metric1,metric1_name))
            metrics_tuples.append((metric2, metric2_name))
        case "balanced":
            # if configs.get("usecase") == 1 or configs.get("usecase") == 2:
            #     metric1 = "binary_accuracy"
            # else:
            #     metric1 = "sparse_categorical_accuracy"
            metric2 = "auc"
            # metric1_name = "Accuracy"
            metric2_name = "AUC"
            # metrics_tuples.append((metric1, metric1_name))
            metrics_tuples.append((metric2, metric2_name))
        case "system":
            metrics_tuples.append(("sent", "Mb send and received"))
            metrics_tuples.append(("total_memory", "Memory usage in MB"))
            metrics_tuples.append(("total_round_time", "Training time in seconds"))
            metrics_tuples.append(
                ("total_per_client_memory_client", "Memory per client in MB")
            )
            metrics_tuples.append(
                ("total_client_time", "Client training time in seconds")
            )

    for metric, metric_name in metrics_tuples:
        if metric == "sent":
            scenario_metrics = mongodb.get_data_by_name(
                f"scenario_metrics_{configs.get('usecase')}_{mode}_network", True
            )
            df1 = create_dfs_for_fl_metric(
                rounds_metrics=scenario_metrics, metric_name="sent"
            )
            df2 = create_dfs_for_fl_metric(
                rounds_metrics=scenario_metrics, metric_name="received"
            )
            plot_swarmplots(
                df1,
                metric_name=metric_name,
                data_path=plot_path,
                plot_type="bar",
                scale=None,
                data2=df2,
                configuration_name="Number of clients",
            )
            continue
        scenario_metrics = mongodb.get_data_by_name(
            f"scenario_metrics_{configs.get('usecase')}_{mode}", True
        )
        df = create_dfs_for_fl_metric(
            rounds_metrics=scenario_metrics, metric_name=metric
        )
        if mode in ["system", "balanced"]:
            central = mongodb.get_data_by_name(
                f"central_metrics_{configs.get('usecase')}_{mode}"
            )
            if mode == "balanced":
                central = {
                    key.split("eval_", 1)[1]: central[key] for key in central.keys()
                }
            if (
                metric == "round_time"
                or metric == "total_client_time"
                or metric == "total_round_time"
            ):
                central_metric = "training_time"
            elif metric == "total_memory" or metric == "total_per_client_memory_client":
                central_metric = "total_memory_central"
            else:
                central_metric = metric
            central_df = transform_to_df(
                central,
                metric_name=central_metric,
                framework="central",
                group="central",
                round_configuration="central",
            )
            df = pd.concat([central_df, df])
        # df_pivot = df.pivot_table(index="group", columns="round configuration", values="metric", aggfunc="mean")
        # scale = [df_pivot.min().min(), df_pivot.max().max()]
        # plot_heatmap(df,"TFF",standard_deviation=False,unweighted=unweighted,metric_name=metric_name,data_path=plot_path,scale=scale)
        # plot_heatmap(df,"FLWR",unweighted=unweighted,scale=scale,metric_name=metric_name,data_path=plot_path)
        # df_pivot = df.pivot_table(index="group", columns="round configuration", values="metric", aggfunc="std")
        # scale = [df_pivot.min().min(), df_pivot.max().max()]
        # plot_heatmap(df,"TFF",standard_deviation=True,unweighted=unweighted,metric_name=metric_name,data_path=plot_path,scale=scale)
        # plot_heatmap(df,"FLWR",standard_deviation=True,unweighted=unweighted,scale=scale,metric_name=metric_name,data_path=plot_path)
        if mode == "unweighted":
            configuration_name = "Percentage of chosen class"
        elif mode == "dp":
            configuration_name = "Noisemultiplier"
        else:
            configuration_name = "Number of clients"
        for plot_type in ["bar"]:
            if metric_name == "AUC":
                scale = [0.0, 1.05]
            else:
                scale = None
            plot_swarmplots(
                df,
                metric_name,
                configuration_name,
                data_path=plot_path,
                plot_type=plot_type,
                scale=scale,
            )
