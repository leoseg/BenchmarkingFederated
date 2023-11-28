from collections import defaultdict

from datapostprocessing.plotting import (
    plot_swarmplots,
    get_metric_names_for_plotting,
    create_legend_patches,
    create_legend_patches_with_sent_received,
)
from datapostprocessing.data_uploading_utils import (
    transform_to_df,
    create_dfs_for_fl_metric,
)
import os
from datapostprocessing.db_utils import MongoDBHandler
import pandas as pd
from matplotlib import pyplot as plt

from datapostprocessing.table_utils import get_metrics_for_mode

"""
Plotting script for which plots all heatmaps and barplots for one usecase
"""
for usecase in [1, 2, 3, 4]:
    os.environ["USECASE"] = str(usecase)
    from config import configs

    mongodb = MongoDBHandler()
    for mode in ["balanced"]:
        plot_path = configs.get("plot_path") + mode + "/"
        unweighted = False
        configuration_name = "Number of clients"
        metrics_tuples = get_metric_names_for_plotting(mode)
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
                elif (
                    metric == "total_memory"
                    or metric == "total_per_client_memory_client"
                ):
                    central_metric = "total_memory_central"
                    df["metric"] = df["metric"] / 1000

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
