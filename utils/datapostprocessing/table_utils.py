from io import StringIO
from typing import Optional

import pandas as pd
import numpy as np
from config import configs

ROUNDS = [1, 3, 5, 10]
if configs.get("usecase") == 2:
    ROUNDS = [1, 2, 4, 8]


def transform_central_metric_to_df(central_data, metric_name):
    """
    Transform central metric to dataframe
    :param central_data: central data to transform
    :param metric_name: metric to transform
    :return: dataframe
    """
    row = [
        {
            "value": f"{np.mean(central_data[metric_name]):.4f} \u00B1 {np.std(central_data[metric_name]):.4f}"
        }
    ]
    df = pd.DataFrame(row)
    return df


def create_summarize_dataframe_from_metrics(
    data: list, metric_name, rounds, groups=None, mode="both"
):
    """
    Create a summarizing dataframe from a list of metrics
    :param data: data to summarize
    :param metric_name: metric to summarize
    :param rounds: rounds to summarize
    :param groups: groups to summarize
    :param mode: mode to summarize
    :return: dataframe
    """
    if groups is None:
        groups = configs.get("groups")
    columns = ["group"] + [f"round {i}" for i in rounds]
    new_columns = []
    for col in columns:
        if col.startswith("round"):
            new_columns.extend([f"{col} tff", f"{col} flwr"])
        else:
            new_columns.append(col)
    df = pd.DataFrame(columns=new_columns)
    for group in groups:
        if group.startswith("tff"):
            row_data = create_data_row(data, group, metric_name, rounds, mode)
            df = df.append(row_data, ignore_index=True)

    # Compute summary columns
    df = compute_summary_columns(df, mode)
    summary_row = create_summary_row(df, mode)
    df = df.append(summary_row, ignore_index=True)
    return df


def create_data_row(data, group, metric_name, rounds, mode):
    """
    Create a data row for a dataframe
    :param data: data to create row for
    :param group: group to create row for
    :param metric_name: metric to create row for
    :param rounds: rounds to create row for
    :param mode: mode to create row for
    :return: row
    """
    row_data = {
        "group": group.split("_")[1],
    }
    for i, round_data in enumerate(data):
        if group.startswith("tff"):
            tff_metrics = round_data[group]
            flwr_group = group.replace("tff", "flwr")
            flwr_metrics = round_data[flwr_group]
            row_data.update(
                {
                    f"round {rounds[i]} tff": create_column_string(
                        np.mean(tff_metrics[metric_name]),
                        np.std(tff_metrics[metric_name]),
                        mode=mode,
                    ),
                    f"round {rounds[i]} flwr": create_column_string(
                        np.mean(flwr_metrics[metric_name]),
                        np.std(flwr_metrics[metric_name]),
                        mode=mode,
                    ),
                }
            )
    return row_data


def create_column_string(mean, std, mode="both"):
    """
    Create a column string
    :param mean: mean
    :param std: std
    :param mode: mode
    :return: string
    """
    if mode == "mean":
        return f"{mean:.4f}"
    elif mode == "std":
        return f"{std:.4f}"
    return f"{mean:.4f} \u00B1 {std:.4f}"


def create_summary_row(df, mode):
    """
    Create a summary row for a dataframe
    :param df: df to create summary row for
    :param mode: mode to create summary row for
    :return: row
    """
    summary_row = {"group": "summary"}
    for col in df.columns:
        if col.startswith("round") or col.startswith("group summary"):
            if mode == "both":
                values = df[col].apply(
                    lambda x: np.array(list(map(float, x.split(" \u00B1 "))))
                )
                mean = np.mean(values.apply(lambda x: x[0]))
                std = np.sqrt(np.sum(values.apply(lambda x: x[1] ** 2)))
            elif mode == "mean":
                mean = np.mean(df[col].apply(lambda x: float(x)))
                std = 0.0
            elif mode == "std":
                std = np.std(df[col].apply(lambda x: float(x)))
                mean = 0.0
            summary_row[col] = create_column_string(mean, std, mode=mode)
    return summary_row


def compute_summary_columns(df, mode):
    """
    Compute summary columns for a dataframe
    :param df: df to compute summary columns for
    :return: df with summary columns
    """
    for index, row in df.iterrows():
        tff_means = []
        flwr_means = []
        tff_stds = []
        flwr_stds = []
        for col in df.columns:
            if col.endswith("tff") and "round" in col:
                if mode == "both":
                    mean, std = map(float, row[col].split(" \u00B1 "))
                    tff_means.append(mean)
                    tff_stds.append(std)
                elif mode == "mean":
                    tff_means.append(float(row[col]))
                    tff_stds.append(0.0)
                elif mode == "std":
                    tff_means.append(0.0)
                    tff_stds.append(float(row[col]))
            elif col.endswith("flwr") and "round" in col:
                if mode == "both":
                    mean, std = map(float, row[col].split(" \u00B1 "))
                    flwr_means.append(mean)
                    flwr_stds.append(std)
                elif mode == "mean":
                    flwr_means.append(float(row[col]))
                    flwr_stds.append(0.0)
                elif mode == "std":
                    flwr_means.append(0.0)
                    flwr_stds.append(float(row[col]))
        if mode == "both":
            flwr_mean = np.mean(flwr_means)
            tff_mean = np.mean(tff_means)
            flwr_std = np.sqrt(np.sum(np.array(flwr_stds) ** 2))
            tff_std = np.sqrt(np.sum(np.array(tff_stds) ** 2))
        elif mode == "mean":
            flwr_mean = np.mean(flwr_means)
            tff_mean = np.mean(tff_means)
            flwr_std = 0
            tff_std = 0
        elif mode == "std":
            flwr_mean = 0
            tff_mean = 0
            flwr_std = np.sqrt(np.sum(np.array(flwr_stds) ** 2))
            tff_std = np.sqrt(np.sum(np.array(tff_stds) ** 2))
        df.at[index, "group summary tff"] = create_column_string(
            tff_mean, tff_std, mode=mode
        )
        df.at[index, "group summary flwr"] = create_column_string(
            flwr_mean, flwr_std, mode=mode
        )
    return df


def transform_df_to_latex(df, headers, rows_to_include, central_df=None):
    """
    Transforms a latex data table to a dataframe with given headers and writes
    it to a text file namd table.txt in the same directory
    :param df: df to transform
    :param headers: headers for the table
    :return:
    """
    df1 = df[["group"] + rows_to_include]
    df2 = df[["group"] + ["group summary tff"] + ["group summary flwr"]]
    latex_str1 = df1.to_latex(index=False, header=headers, escape=False)
    latex_str2 = df2.to_latex(
        index=False,
        header=["Number of Clients", "Group summary TFF", "Group summary Flwr"],
        escape=False,
    )
    latex_str3 = ""

    if central_df is not None:
        latex_str3 = central_df.to_latex(
            index=False, header=["Central model value"], escape=False
        )
    complete_latex_str = latex_str1 + "\n" + latex_str2 + "\n" + latex_str3
    with open("table.txt", "w") as f:
        f.write(complete_latex_str)


def transform_df_to_landscape_table(df, headers, caption, central_df=None):
    """
    Transforms a latex data table to a dataframe with given headers and writes jt to file
    :param df: df to transform
    :param headers: headers for the table
    :return:
    """
    df_string = df.to_latex(index=False, header=headers, escape=False)
    central_df_string = ""
    if central_df is not None:
        central_df_string = central_df.to_latex(
            index=False, header=["Central model value"], escape=False
        )
    return (
        "\\begin{table}\n \caption{"
        + caption
        + "}\n \\begin{tiny}\n "
        + df_string
        + central_df_string
        + "\n\end{tiny}\n\end{table}\n"
    )
    # with open("landscape_table.txt", "w") as f:
    #     f.write(df_string)


def get_usecase_name(usecase: int):
    """
    Returns the name of the usecase
    :param usecase: usecase number
    :return: name of the usecase
    """
    match usecase:
        case 1:
            return "BloodDL"
        case 2:
            return "BloodLog"
        case 3:
            return "BrainCellLog"
        case 4:
            return "BrainCellDL"


def get_mode_name(mode: str):
    """
    Returns the name of the mode
    :param mode: mode
    :return: name of the mode
    """
    match mode:

        case "balanced":
            return "Benchmark model performance for number of clients"
        case "unweighted":
            return "Benchmark model performance for class imbalance"
        case "system":
            return "Benchmark computational resources"


def get_metrics_for_mode(mode, usecase):
    """
    Returns the metrics for a given mode
    :param mode: mode to return metrics for
    :return: list of metrics
    """
    match mode:
        case "balanced":
            if usecase == 1 or usecase == 2:
                metric1 = "binary_accuracy"
            else:
                metric1 = "sparse_categorical_accuracy"
            return [
                ("auc", "AUC"),
                (metric1, "Accuracy"),
            ]
        case "unweighted":
            if usecase == 1 or usecase == 2:
                metric1 = "binary_accuracy_global"
            else:
                metric1 = "sparse_categorical_accuracy_global"
            return [
                ("auc_global", "AUC"),
                (metric1, "Accuracy"),
            ]
        case "system":
            return [
                ("total_memory", "Total memory usage in MB"),
                ("total_round_time", "Training time in seconds of all rounds"),
                ("total_per_client_memory_client", "Memory per client in MB"),
                ("total_client_time", "Training time in seconds per client"),
                ("sent", "Mb sent from client"),
                ("received", "Mb received by client"),
            ]


def convert_val(val):
    """
    Converts a value to a string with the correct number of significant digits
    :param val: value to convert
    :return: string with correct number of significant digits
    """
    parts = val.split(" ± ")

    # Convert each part to float, round to nearest integer, convert back to string
    parts = [str(int(round(float(part)))) for part in parts]

    # Concatenate the parts back together
    return " ± ".join(parts)
