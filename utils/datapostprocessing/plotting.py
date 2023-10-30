import pandas as pd
import seaborn as sns
from config import configs
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

ENTITY = "Scads"
if configs.get("usecase") == 2:
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


def seaborn_plot(
    x,
    metric_name,
    hue,
    data,
    palette,
    title,
    data_path,
    dodge=True,
    configuration_name="Number of Rounds",
    plot_type="box",
    scale=None,
    data2=None,
):
    """
    Plots a seaborn boxplot to easy exchange plottype
    :param x: x axis data
    :param metric_name: metric nane
    :param hue: hue data
    :param data: data to plot
    :param palette: color palette
    :param title: title of the plot
    :param dodge: dodge parameter
    :param configuration_name: name of the configuration
    :param save_fig_ext: extension for saving the figure
    :param data_path: path to the data
    :return:
    """
    sns.set(font_scale=1.7)
    plt.subplots_adjust(bottom=0.24, left=0.17)

    sns_palette = sns.color_palette("Set2")
    palette = {
        "TFF": sns_palette[1],
        "Centralized": sns_palette[0],
        "FLWR": sns_palette[2],
    }
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
                ax = sns.barplot(x=x, y="metric", hue=hue, data=data2, palette=palette2)
                sns.barplot(
                    x=x,
                    y="metric",
                    hue=hue,
                    data=data,
                    palette=palette,
                    dodge=dodge,
                    ax=ax,
                )
                bottom_bar_tff = mpatches.Patch(
                    color=sns_palette2[1], label="TFF Received"
                )
                bottom_bar_flwr = mpatches.Patch(
                    color=sns_palette2[2], label="FLWR Received"
                )
                top_bar_tff = mpatches.Patch(color=sns_palette[1], label="TFF Sent")
                top_bar_flwr = mpatches.Patch(color=sns_palette[2], label="FLWR Sent")
                # for g in ax.patches:
                #     ax.annotate(format(g.get_height(), '.1f'),
                #                    (g.get_x() + g.get_width() / 2., g.get_height()),
                #                    ha='center', va='center',
                #                    xytext=(0, 9),
                #                    textcoords='offset points')
                plt.legend(
                    handles=[top_bar_tff, bottom_bar_tff, top_bar_flwr, bottom_bar_flwr]
                )
            else:
                ax = sns.barplot(
                    x=x, y="metric", hue=hue, data=data, palette=palette, dodge=dodge
                )

        case "box_points":
            ax = sns.boxplot(
                x=x, y="metric", hue=hue, data=data, palette=palette, dodge=dodge
            )
            sns.stripplot(data=data, x=x, y="metric", hue=hue, dodge=True)
    if metric_name != "Mb send and received":
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
    plt.ylabel(metric_name)
    if metric_name == "AUC":
        plt.axhline(0.5, color="red", linestyle="--")
    if scale is not None:
        plt.ylim(scale[0], scale[1])
    # set x axis title to group name
    plt.xlabel(configuration_name)
    if configuration_name == "Percentage of chosen class":
        if configs.get("usecase") == 1 or configs.get("usecase") == 2:
            start_value = 50
        else:
            start_value = 20
        x_ticks = [
            (int(float(x)) * 5 + start_value)
            for x in data[x].unique()
            if x != "central"
        ]
        ax.set_xticklabels(x_ticks)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plt.savefig(data_path + metric_name + "_" + title + plot_type + ".png")
    plt.show()


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
):
    """
    Plots swarmplots for the given df which contains all data for one metric
    :param df: df with all data for one metric
    :param metric_name: metric name
    :param configuration_name: configuration name
    :param data_path: path to the data
    :metric_name: name of metric to plot
    """
    if metric_name == "AUC":
        for index, round_num in enumerate(ROUNDS):
            if round_num in [1, 10]:
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
                    "framework",
                    round_df,
                    "Set2",
                    f"Round configuration {round_num}",
                    configuration_name=configuration_name,
                    plot_type=plot_type,
                    data_path=data_path,
                    scale=scale,
                    data2=round_data2,
                )
                plt.show()
    seaborn_plot(
        "group",
        metric_name,
        "framework",
        df,
        "Set2",
        f"Group configs over all round configs",
        configuration_name=configuration_name,
        plot_type=plot_type,
        data_path=data_path,
        scale=scale,
        data2=data2,
    )
    seaborn_plot(
        "round configuration",
        metric_name,
        "framework",
        df,
        "Set2",
        f"Round configs over all groups",
        plot_type=plot_type,
        data_path=data_path,
        scale=scale,
        data2=data2,
    )
    if metric_name == "Mb send and received":
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
                "framework",
                group_df,
                "Set2",
                f"Group {group}",
                plot_type=plot_type,
                data_path=data_path,
                scale=scale,
                data2=group_data2,
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
