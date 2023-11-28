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
plot_dict = defaultdict()
plots_refs = defaultdict
position_dict = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}
modes = ["system", "balanced", "unweighted", "dp"]
for mode in modes:
    metric_names = [
        metric_tuple[0] for metric_tuple in get_metric_names_for_plotting(mode)
    ]
    for metric_name in metric_names:
        fig_rounds, ax_rounds = plt.subplots(2, 2, figsize=(10, 10))
        fig_groups, ax_groups = plt.subplots(2, 2, figsize=(10, 10))
        plots_refs[mode][metric_name]["rounds"] = ax_rounds, fig_rounds
        plots_refs[mode][metric_name]["groups"] = ax_groups, fig_groups
        for usecase in [1, 2, 3, 4]:
            plot_dict[mode][metric_name][usecase]["rounds"] = ax_rounds[
                position_dict[usecase]
            ]
            plot_dict[mode][metric_name][usecase]["groups"] = ax_groups[
                position_dict[usecase]
            ]


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
                    axs_object=plot_dict[mode],
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
                print(f"Object id for {usecase} {metric} rounds")
                print(id(plot_dict[metric][usecase]["rounds"]))
                plot_swarmplots(
                    df,
                    metric_name,
                    configuration_name,
                    data_path=plot_path,
                    plot_type=plot_type,
                    scale=scale,
                    axs_object=plot_dict[mode][metric][usecase],
                )


for mode in modes:
    for metric_name in plots_refs[mode].keys():
        for summarize_mode in ["rounds", "groups"]:
            print(f"object id for {metric_name} {summarize_mode} ")
            print(id(plots_refs[mode][metric_name][summarize_mode][0][0][0]))
            axs, fig = plots_refs[mode][metric_name][summarize_mode]
            axs[0, 0].set_title("a")
            axs[0, 1].set_title("b")
            axs[1, 0].set_title("c")
            axs[1, 1].set_title("d")
            if metric_name == "sent":
                legend_patches = create_legend_patches_with_sent_received()
            elif mode in ["system", "balanced"]:
                legend_patches = create_legend_patches(True)
            else:
                legend_patches = create_legend_patches()
            fig.legend(handles=legend_patches, loc="lower center")
            plt.tight_layout()
            plt.savefig(
                f"../../BenchmarkData/plots/plot_{metric_name}_{summarize_mode}.png"
            )
