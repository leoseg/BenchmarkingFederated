from typing import Optional

import pandas as pd
import numpy as np
from config import configs
ROUNDS = [1,3,5,10]
if configs.get("usecase") == 2:
    ROUNDS = [1,2,4,8]

def transform_central_metric_to_df(central_data,metric_name):
    """
    Transform central metric to dataframe
    :param central_data: central data to transform
    :param metric_name: metric to transform
    :return: dataframe
    """
    row = [{"value": f"{np.mean(central_data[metric_name]):.4f} \u00B1 {np.std(central_data[metric_name]):.4f}"}]
    df = pd.DataFrame(row)
    return df
def create_summarize_dataframe_from_metrics(data: list,metric_name, rounds,groups = None,):
    """
    Create a summarizing dataframe from a list of metrics
    :param data: data to summarize
    :param metric_name: metric to summarize
    :param rounds: rounds to summarize
    :param groups: groups to summarize
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
            row_data = create_data_row(data, group, metric_name, rounds)
            df = df.append(row_data, ignore_index=True)

    # Compute summary columns
    df = compute_summary_columns(df)
    summary_row = create_summary_row(df)
    df = df.append(summary_row, ignore_index=True)
    return df


def create_data_row(data, group, metric_name, rounds):
    """
    Create a data row for a dataframe
    :param data: data to create row for
    :param group: group to create row for
    :param metric_name: metric to create row for
    :param rounds: rounds to create row for
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
            row_data.update({
                f"round {rounds[i]} tff": f"{np.mean(tff_metrics[metric_name]):.4f} \u00B1 {np.std(tff_metrics[metric_name]):.4f}",
                f"round {rounds[i]} flwr": f"{np.mean(flwr_metrics[metric_name]):.4f} \u00B1 {np.std(flwr_metrics[metric_name]):.4f}",
            })
    return row_data


def create_summary_row(df):
    """
    Create a summary row for a dataframe
    :param df: df to create summary row for
    :return: row
    """
    summary_row = {"group": "summary"}
    for col in df.columns:
        if col.startswith("round") or col.startswith("group summary"):
            values = df[col].apply(lambda x: np.array(list(map(float, x.split(" \u00B1 ")))))
            mean = np.mean(values.apply(lambda x: x[0]))
            std = np.sqrt(np.sum(values.apply(lambda x: x[1] ** 2)))
            summary_row[col] = f"{mean:.4f} \u00B1 {std:.4f}"
    return summary_row


def compute_summary_columns(df,stdv=True):
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
                mean, std = map(float, row[col].split(" \u00B1 "))
                tff_means.append(mean)
                tff_stds.append(std)
            elif col.endswith("flwr") and "round" in col:
                mean, std = map(float, row[col].split(" \u00B1 "))
                flwr_means.append(mean)
                flwr_stds.append(std)
        column_value_tff = f"{np.mean(tff_means):.4f}"
        column_value_flwr = f"{np.mean(flwr_means):.4f}"
        if stdv:
            column_value_flwr = column_value_flwr + f" \u00B1 {np.sqrt(np.sum(np.array(flwr_stds) ** 2)):.4f}"
            column_value_tff = column_value_tff + f" \u00B1 {np.sqrt(np.sum(np.array(tff_stds) ** 2)):.4f}"
        df.at[
            index, "group summary tff"] = column_value_tff
        df.at[
            index, "group summary flwr"] = column_value_flwr
    return df


def transform_df_to_latex(df,headers,rows_to_include,central_df=None):
    """
    Transforms a latex data table to a dataframe with given headers and writes
    it to a text file namd table.txt in the same directory
    :param df: df to transform
    :param headers: headers for the table
    :return:
    """
    df1 = df[["group"] + rows_to_include]
    df2 = df[["group"] + ["group summary tff"]+["group summary flwr"]]
    latex_str1 = df1.to_latex(index=False, header=headers, escape=False)
    latex_str2 = df2.to_latex(index=False, header=["Number of Clients","Group summary TFF","Group summary Flwr"], escape=False)
    latex_str3 = ""

    if central_df is not None:
        latex_str3 = central_df.to_latex(index=False, header=["Central model value"], escape=False)
    complete_latex_str= latex_str1 + "\n" + latex_str2 + "\n"+  latex_str3
    with open("table.txt", "w") as f:
        f.write(complete_latex_str)


def transform_df_to_landscape_table(df,headers,caption, central_df=None):
    """
    Transforms a latex data table to a dataframe with given headers and writes jt to file
    :param df: df to transform
    :param headers: headers for the table
    :return:
    """
    df_string = df.to_latex(index=False, header=headers, escape=False)
    central_df_string =""
    if central_df is not None:
        central_df_string = central_df.to_latex(index=False, header=["Central model value"], escape=False)
    df_string = "\\begin{landscape}\n \\begin{table}\n \caption{"+caption+ "}\n \\begin{tiny}\n " + df_string + central_df_string+"\n\end{tiny}\n\end{table}\n\end{landscape}"
    with open("landscape_table.txt", "w") as f:
        f.write(df_string)





