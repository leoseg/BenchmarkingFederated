import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from config import configs, get_config
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches


ENTITY = "Scads"
if get_config().get("usecase") == 2:
    ROUNDS = [1, 2, 4, 8]
else:
    ROUNDS = [1, 2, 5, 10]

plt.rcParams["axes.formatter.limits"] = [-5, 3]


def create_loss_line_plot(df, plot_path: str):
    """
    Creates a lineplot for the loss
    :param df: df with loss values
    :param plot_path: path to save plot
    :return:
    """
    ax = sns.lineplot(df, x="round", y="loss", hue="framework")
    plt.xlabel("Round(FL)/Epoch(central)")
    num_rounds = 10
    if configs.get("usecase") == 2:
        num_rounds = 8
    ticks = [
        f"{element}/{int(element * configs.get('epochs') / num_rounds)}"
        for element in df["round"].unique().tolist()
    ]
    ax.set_xticks(df["round"].unique().tolist())
    ax.set_xticklabels(ticks)
    plt.savefig(plot_path + "loss.png")
    plt.show()


def create_legend_patches(with_centralized=False):
    """
    Creates patches for legend
    """
    sns_palette = sns.color_palette("Set2")
    tff_patch = mpatches.Patch(color=sns_palette[1], label="TFF")
    flwr_patch = mpatches.Patch(color=sns_palette[2], label="FLWR")
    patches = [tff_patch, flwr_patch]
    if with_centralized:
        centralized_patch = mpatches.Patch(color=sns_palette[0], label="Centralized")
        patches.append(centralized_patch)
    return patches


def create_legend_patches_with_sent_received():
    """
    Creates patches for legend with send and received
    """
    sns_palette = sns.color_palette("Set2")
    sns_palette2 = sns.color_palette("Dark2")
    bottom_bar_tff = mpatches.Patch(color=sns_palette2[1], label="TFF Received")
    bottom_bar_flwr = mpatches.Patch(color=sns_palette2[2], label="FLWR Received")

    top_bar_tff = mpatches.Patch(color=sns_palette[1], label="TFF Sent")
    top_bar_flwr = mpatches.Patch(color=sns_palette[2], label="FLWR Sent")
    patches = [top_bar_tff, bottom_bar_tff, top_bar_flwr, bottom_bar_flwr]
    return patches


def seaborn_plot(
    x,
    metric_name,
    data,
    title,
    data_path,
    configuration_name="Number of rounds",
    plot_type="box",
    scale=None,
    data2=None,
    axs_object=None,
):
    """
    Plots a seaborn boxplot to easy exchange plottype
    :param x: x axis data
    :param metric_name: metric nane
    :param data: data to plot
    :param title: title of the plot
    :param configuration_name: name of the configuration
    :param save_fig_ext: extension for saving the figure
    :param data_path: path to the data
    :return:
    """
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    # plt.subplots_adjust(bottom=0.24, left=0.17)
    hue = "framework"
    dodge = True

    sns_palette = sns.color_palette("Set2")
    palette = {
        "TFF": sns_palette[1],
        "Centralized": sns_palette[0],
        "FLWR": sns_palette[2],
    }
    if axs_object is not None:
        legend = False
        print(f"{id(axs_object)} for {configuration_name}")
    else:
        legend = True
    match plot_type:
        case "box":
            ax = sns.boxplot(
                x=x, y="metric", hue=hue, data=data, palette=palette, dodge=dodge
            )
        case "bar":
            if data2 is not None:
                data2.reset_index(inplace=True)
                data2.drop(columns=["index"], inplace=True)
                data.reset_index(inplace=True)
                data.drop(columns=["index"], inplace=True)
                data2["metric"] = (data2["metric"] + data["metric"]).tolist()
                sns_palette2 = sns.color_palette("Dark2")
                palette2 = {"TFF": sns_palette2[1], "FLWR": sns_palette2[2]}
                sns.set_style("whitegrid")
                ax = sns.barplot(
                    x=x,
                    y="metric",
                    hue=hue,
                    data=data2,
                    palette=palette2,
                    legend=legend,
                    ax=axs_object,
                )
                sns.barplot(
                    x=x,
                    y="metric",
                    hue=hue,
                    data=data,
                    palette=palette,
                    dodge=dodge,
                    ax=ax,
                    legend=legend,
                )

                if legend:
                    legends = create_legend_patches_with_sent_received()
                    plt.legend(handles=legends)
            else:
                ax = sns.barplot(
                    x=x,
                    y="metric",
                    hue=hue,
                    data=data,
                    palette=palette,
                    dodge=dodge,
                    legend=legend,
                    ax=axs_object,
                )

        case "box_points":
            ax = sns.boxplot(
                x=x, y="metric", hue=hue, data=data, palette=palette, dodge=dodge
            )
            sns.stripplot(data=data, x=x, y="metric", hue=hue, dodge=True)
    if metric_name != "Network traffic (MB)" and legend:
        sns.move_legend(
            ax,
            "upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
            title=None,
            frameon=False,
        )
    # ax.set_title(title)
    # set y axis title to metric name
    ax.grid(visible=True, which="major", axis="y")
    ax.set_axisbelow(True)
    fontsize = 15
    ax.set_ylabel(metric_name, fontsize=fontsize)
    if metric_name == "AUC":
        ax.axhline(0.5, color="red", linestyle="--")
    if "Memory" in metric_name:

        def custom_formatter(x, pos):
            return f"{int(x)}"

        # Set the custom formatter for the y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    if scale is not None:
        ax.set_ylim(scale[0], scale[1])
    # set x axis title to group name
    ax.set_xlabel(configuration_name, fontsize=fontsize)
    if configuration_name == "Imbalance (%)":
        if os.environ["USECASE"] == str(1) or os.environ["USECASE"] == str(2):
            start_value = 50
        else:
            start_value = 20
        x_ticks = [
            (int(float(x)) * 5 + start_value)
            for x in data[x].unique()
            if x != "central"
        ]
        ax.set_xticklabels(x_ticks)
    ax.tick_params(axis="both", labelsize=13)
    ax.tick_params(axis="x", length=0)
    if axs_object is None:
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        plt.savefig(data_path + metric_name + "_" + title + plot_type + ".pdf")
        plt.show()
    return ax


def plot_heatmap(
    df,
    framework,
    metric_name,
    data_path,
    standard_deviation=False,
    unweighted=False,
    scale=None,
):
    """
    Plots a heatmap for the given df which contains all data for one metric
    :param df: df with all data for one metric
    :param framework: framework name
    :param standard_deviation: if true, the standard deviation is plotted instead of the mean
    :param unweighted: if true, the unweighted usecase is plotted
    :param scale: scale for the heatmap
    :param data_path: path to the data
    :return: returns scale so it can be used for latter plots
    """

    df = df[df["framework"] == framework]
    if unweighted:
        df["group"] = df["group"].astype(float)
    df["group"] = df["group"].astype(int)
    if standard_deviation:
        type_of = "stdv"
        title = (
            "Standard deviation of " + metric_name + " over all repeats of " + framework
        )
    else:
        title = "Mean " + metric_name + " over all repeats of " + framework
        type_of = "mean"

    # plt.title(title)
    if standard_deviation:
        df = df.pivot_table(
            index="group",
            columns="round configuration",
            values="metric",
            aggfunc="std",
            margins=True,
        )
    else:
        df = df.pivot_table(
            index="group", columns="round configuration", values="metric", margins=True
        )

    # df = df.pivot("group", "round configuration", "metric")
    if scale is None:
        scale = [df.min().min(), df.max().max()]
    ax = sns.heatmap(df, cmap="rocket_r", vmin=scale[0], vmax=scale[1], annot=True)
    y_title = "Number of clients"
    x_title = "Number of rounds trained"
    if unweighted:
        y_title = "Percentage of chosen class"
        if configs.get("usecase") == 1 or configs.get("usecase") == 2:
            start_value = 50
        else:
            start_value = 20
        y_ticks = [
            (int(float(x)) * 5 + start_value)
            for x in df.index
            if x != "central" and x != "All"
        ]
        y_ticks.append("All")
        ax.set_yticklabels(y_ticks)
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plt.savefig(f"{data_path}{metric_name}_{framework}_{type_of}.png")
    plt.show()
    return scale


def calc_scale(df, mode):
    if mode != "system":
        max_value = 1.0
        min_value = min(df["metric"].min() - 0.1, 0)
    else:
        max_value = df["metric"].max() + 0.7 * df["metric"].max()
        min_value = min(df["metric"].min() - 0.7 * df["metric"].min(), 0)
    return (min_value, max_value)


def plot_swarmplots(
    df,
    metric_name: str,
    configuration_name: str,
    data_path: str,
    plot_type: str,
    scale=None,
    data2=None,
    axs_object=None,
):
    """
    Plots swarmplots for the given df which contains all data for one metric
    :param df: df with all data for one metric
    :param metric_name: metric name
    :param configuration_name: configuration name
    :param data_path: path to the data
    :metric_name: name of metric to plot
    """
    # print(id(axs_object["rounds"]))
    # print(" id for axs object groups")
    # print(id(axs_object["groups"]))
    if metric_name == "Network traffic (MB)":
        df = df[df["group"] == "10"]
        for group in df["group"].unique():

            if group == "central":
                continue
            group_df = df[df["group"] == "central"]
            group_df = pd.concat(
                [group_df, df[df["group"] == group]], ignore_index=True
            )
            if data2 is not None:
                group_data2 = data2[data2["group"] == group]
            else:
                group_data2 = None
            seaborn_plot(
                "round configuration",
                metric_name,
                group_df,
                f"Group {group}",
                plot_type=plot_type,
                data_path=data_path,
                scale=scale,
                data2=group_data2,
                axs_object=axs_object["rounds"],
            )
    else:
        seaborn_plot(
            "round configuration",
            metric_name,
            df,
            f"Round configs over all groups",
            plot_type=plot_type,
            data_path=data_path,
            scale=scale,
            data2=data2,
            axs_object=axs_object["rounds"],
        )
        seaborn_plot(
            "group",
            metric_name,
            df,
            f"Group configs over all round configs",
            configuration_name=configuration_name,
            plot_type=plot_type,
            data_path=data_path,
            scale=scale,
            data2=data2,
            axs_object=axs_object["groups"],
        )


def plot_round(
    df,
    metric_name: str,
    configuration_name: str,
    data_path: str,
    plot_type: str,
    round_to_plot: int,
    scale=None,
    data2=None,
    axs_object=None,
):
    """
    Plots all groups summarized for the given round
    """
    for index, round_num in enumerate(ROUNDS):
        if round_num == round_to_plot:
            round_df = df[df["round configuration"] == "central"]
            round_df = pd.concat(
                [round_df, df[df["round configuration"] == round_num]],
                ignore_index=True,
            )
            if data2 is not None:
                round_data2 = data2[data2["round configuration"] == round_num]
            else:
                round_data2 = None
            seaborn_plot(
                "group",
                metric_name,
                round_df,
                f"Round configuration {round_num}",
                configuration_name=configuration_name,
                plot_type=plot_type,
                data_path=data_path,
                scale=scale,
                data2=round_data2,
                axs_object=axs_object[round_to_plot],
            )


def create_time_diff(df1, df2, relation=False):
    """
    Creates a df with the time difference between df1 and df2
    :param df1: df with round time
    :param df2: df with client time
    :param relation: if true, return the factor round time is longer than client time
    :return:
    """
    if relation:
        df1["metric"] = df1["metric"] / df2["metric"]
        return df1
    round_time = df1["metric"]
    df1["metric"] = df1["metric"] - df2["metric"]

    df1["metric"] = df1["metric"] / df1["round configuration"]
    return df1


def get_metric_names_for_plotting(mode) -> list:
    """
    Returns the metric names for the given mode
    """
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
            # metrics_tuples.append((metric1,metric1_name))
            metrics_tuples.append((metric2, metric2_name))
            configuration_name = "Imbalance (%)"
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
        case "1vs10":
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
            metrics_tuples.append(("sent", "Network traffic (MB)"))
            metrics_tuples.append(("total_memory", "Memory (GB)"))
            metrics_tuples.append(("total_round_time", "Training time (s)"))
            metrics_tuples.append(("total_per_client_memory_client", "Memory(GB)"))
            metrics_tuples.append(("total_client_time", "Training time (s)"))
            metrics_tuples.append(("max_per_client_memory_client", "Max Memory (GB)"))
            metrics_tuples.append(("mean_per_client_memory_client", "Mean Memory (GB)"))
        case "dp":
            configuration_name = "Noise multiplier"
            metrics_tuples.append(("auc", "AUC"))
    return metrics_tuples


def plot_figure_with_subfigures(axs, fig, mode, metric_name, summarize_mode):
    """
    Plots the figure with subfigures
    """
    axs[0, 0].set_title("a)")
    axs[0, 1].set_title("b)")
    axs[1, 0].set_title("c)")
    axs[1, 1].set_title("d)")
    if metric_name == "sent":
        legend_patches = create_legend_patches_with_sent_received()
    elif mode in ["system", "balanced", "1vs10"]:
        legend_patches = create_legend_patches(True)
    else:
        legend_patches = create_legend_patches()
    fig.legend(
        handles=legend_patches, bbox_to_anchor=(0.5, 0.05), loc="lower center", ncol=3
    )  # ,bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(f"../../BenchmarkData/plots/{mode}_{metric_name}_{summarize_mode}.pdf")
