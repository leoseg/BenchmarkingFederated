import os
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt

from config import configs
from datapostprocessing.data_uploading_utils import (
    create_dfs_for_fl_metric,
    transform_to_df,
)
from datapostprocessing.db_utils import MongoDBHandler
from datapostprocessing.plotting import (
    get_metric_names_for_plotting,
    plot_swarmplots,
    plot_figure_with_subfigures,
    plot_round,
)

modes = ["system"]

data = {}
mongodb = MongoDBHandler()
network_data = {}
central_data = {}
for mode in modes:
    data[mode] = {}
    central_data[mode] = {}
    for usecase in [1, 2, 3, 4]:
        os.environ["USECASE"] = str(usecase)
        if mode == "1vs10":
            data[mode][usecase] = mongodb.get_data_by_name(
                f"scenario_metrics_{usecase}_balanced", True
            )

        else:
            data[mode][usecase] = mongodb.get_data_by_name(
                f"scenario_metrics_{usecase}_{mode}", True
            )
        if mode == "system":
            network_data[usecase] = mongodb.get_data_by_name(
                f"scenario_metrics_{usecase}_{mode}_network", True
            )
        if mode in ["system", "balanced", "1vs10"]:
            if mode == "1vs10":
                central = mongodb.get_data_by_name(
                    f"central_metrics_{usecase}_balanced"
                )
            else:
                central = mongodb.get_data_by_name(f"central_metrics_{usecase}_{mode}")
            if mode in ["balanced", "1vs10"]:
                central = {
                    key.split("eval_", 1)[1]: central[key] for key in central.keys()
                }
            central_data[mode][usecase] = central


for mode in modes:
    metrics_tuples = get_metric_names_for_plotting(mode)
    if mode == "unweighted":
        configuration_name = "Imbalance (%)"
    elif mode == "dp":
        configuration_name = "Noise"
    else:
        configuration_name = "Number of clients"

    for metric, metric_name in metrics_tuples:

        fig_rounds, ax_rounds = plt.subplots(2, 2, figsize=(10, 10))
        fig_groups, ax_groups = plt.subplots(2, 2, figsize=(10, 10))
        if mode == "1vs10":
            axs_dict = {
                1: {1: ax_groups[0, 0], 10: ax_groups[0, 1]},
                4: {1: ax_groups[1, 0], 10: ax_groups[1, 1]},
            }
        else:
            axs_dict = {
                1: {
                    "rounds": ax_rounds[0, 0],
                    "groups": ax_groups[0, 0],
                },
                2: {
                    "rounds": ax_rounds[1, 0],
                    "groups": ax_groups[1, 0],
                },
                3: {
                    "rounds": ax_rounds[1, 1],
                    "groups": ax_groups[1, 1],
                },
                4: {
                    "rounds": ax_rounds[0, 1],
                    "groups": ax_groups[0, 1],
                },
            }
        for usecase in [1, 2, 3, 4]:
            if usecase == 2:
                rounds = [1, 2, 4, 8]
            else:
                rounds = [1, 2, 5, 10]
            os.environ["USECASE"] = str(usecase)
            scenario_metrics = data[mode][usecase]
            if metric == "sent":
                scenario_metrics = network_data[usecase]
                df1 = create_dfs_for_fl_metric(
                    rounds_metrics=scenario_metrics, metric_name="sent", rounds=rounds
                )
                df2 = create_dfs_for_fl_metric(
                    rounds_metrics=scenario_metrics,
                    metric_name="received",
                    rounds=rounds,
                )
                print(f" id from axs dict for usecase {usecase} for rounds")
                print(id(axs_dict[usecase]["rounds"]))
                print(f" id from axs dict for usecase {usecase} for groups")
                print(id(axs_dict[usecase]["groups"]))
                plot_swarmplots(
                    df1,
                    metric_name=metric_name,
                    data_path="",
                    plot_type="bar",
                    scale=None,
                    data2=df2,
                    configuration_name=configuration_name,
                    axs_object=axs_dict[usecase],
                )
                continue
            df = create_dfs_for_fl_metric(
                rounds_metrics=scenario_metrics, metric_name=metric, rounds=rounds
            )
            if mode in ["system", "balanced", "1vs10"]:
                central = central_data[mode][usecase]
                if (
                    metric == "round_time"
                    or metric == "total_client_time"
                    or metric == "total_round_time"
                ):
                    central_metric = "training_time"
                elif metric in ["total_memory", "total_per_client_memory_client"]:
                    central_metric = "total_memory_central"
                elif metric == "max_per_client_memory_client":
                    central_metric = "max_per_client_memory_central"
                elif metric == "mean_per_client_memory_client":
                    central_metric = "mean_per_client_memory_central"
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
                if metric in [
                    "total_memory",
                    "total_per_client_memory_client",
                    "max_per_client_memory_client",
                    "mean_per_client_memory_client",
                ]:
                    df["metric"] = df["metric"] / 1000
            if metric_name == "AUC":
                scale = [0.0, 1.05]
            else:
                scale = None

            if mode == "1vs10":
                if usecase in [1, 4]:
                    plot_round(
                        df,
                        metric_name=metric_name,
                        data_path="",
                        plot_type="bar",
                        scale=scale,
                        configuration_name=configuration_name,
                        axs_object=axs_dict[usecase],
                        round_to_plot=1,
                    )
                    plot_round(
                        df,
                        metric_name=metric_name,
                        data_path="",
                        plot_type="bar",
                        scale=scale,
                        configuration_name=configuration_name,
                        axs_object=axs_dict[usecase],
                        round_to_plot=10,
                    )
            else:
                #
                # print(f" id from axs dict for usecase {usecase} for rounds")
                # print(id(axs_dict[usecase]["rounds"]))
                # print(f" id from axs dict for usecase {usecase} for groups")
                # print(id(axs_dict[usecase]["groups"]))
                # print(df.tail())
                if mode == "dp":
                    df = df.sort_values("group", ascending=True)
                plot_swarmplots(
                    df,
                    metric_name=metric_name,
                    data_path="",
                    plot_type="bar",
                    scale=scale,
                    configuration_name=configuration_name,
                    axs_object=axs_dict[usecase],
                )

        plot_figure_with_subfigures(ax_groups, fig_groups, mode, metric, "groups")
        plot_figure_with_subfigures(ax_rounds, fig_rounds, mode, metric, "rounds")
