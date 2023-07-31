import pandas as pd

from table_utils import transform_df_to_latex, create_summarize_dataframe_from_metrics, transform_central_metric_to_df, \
    transform_df_to_landscape_table, get_usecase_name, get_metrics_for_mode, get_mode_name, convert_val
from db_utils import MongoDBHandler

headers = ["Nr. Clients","1 Rounds TFF", " 1 Rounds Flwr","3 Rounds TFF", "3 Rounds Flwr","5 Rounds TFF", "5 Rounds Flwr","10 Rounds TFF", "10 Rounds Flwr", "Rounds Summary TFF", "Rounds Summary Flwr"]
mongodb = MongoDBHandler()
appendixstr = ""
for usecase in [1,2,3,4]:
    appendixstr += f"\chapter{{Usecase {usecase}}}\n"
    for mode in ["system","balanced","unweighted"]:

        appendixstr += f"\section{{{get_mode_name(mode)}}}\n"
        if mode == "unweighted":
            if usecase in [1,2]:
                groups =["tff_0.0","tff_2.0","tff_4.0","tff_6.0","tff_8.0","tff_9.0","tff_10.0",
                             "flwr_0.0","flwr_2.0","flwr_4.0","flwr_6.0","flwr_8.0","flwr_9.0","flwr_10.0"]
            else:
                groups = ["tff_0.0", "tff_4.0", "tff_8.0", "tff_10.0", "tff_12.0", "tff_14.0","tff_16.0",
                           "flwr_0.0", "flwr_4.0", "flwr_8.0", "flwr_10.0", "flwr_12.0", "flwr_14.0","flwr_16.0"]
        else:
            groups = ["tff_3","flwr_3","tff_5","flwr_5","tff_10","flwr_10","tff_50","flwr_50"]
        data = mongodb.get_data_by_name(f"scenario_metrics_{usecase}_{mode}",calc_total_memory=True)
        appendixstr += "\\begin{landscape}\n "
        for metric,metric_name in get_metrics_for_mode(mode,usecase):
            if metric in ["sent","received"]:
                data2 = mongodb.get_data_by_name(f"scenario_metrics_{usecase}_system_network", calc_total_memory=True)
                df = create_summarize_dataframe_from_metrics(data=data2,metric_name=metric,rounds=[1,3,5,10],groups=groups)
                central_df = None
            else:
                df = create_summarize_dataframe_from_metrics(data=data,metric_name=metric,rounds=[1,3,5,10],groups=groups)
            if mode == "unweighted" or metric in ["sent","received"]:
                central_df = None
            else:
                central = mongodb.get_data_by_name(f"central_metrics_{usecase}_{mode}")
                if mode == "balanced":
                    central = {key.split("eval_", 1)[1]: central[key] for key in central.keys()}
                if metric == "round_time" or metric == "total_client_time" or metric == "total_round_time":
                    central_metric = "training_time"
                elif metric == "total_memory" or metric == "total_per_client_memory_client":
                    central_metric = "total_memory_central"
                else:
                    central_metric = metric
                central_df = transform_central_metric_to_df(central, metric_name=central_metric)
            if "memory" in metric:
                cols = [col for col in df.columns if col != 'group']
                for col in cols:
                    df[col] = df[col].apply(convert_val)
            appendixstr += transform_df_to_landscape_table(df,headers, f"{metric_name} for {get_usecase_name(usecase)}",central_df=central_df)
        appendixstr += "\end{landscape}"
        appendixstr += "\\newpage"
with open("table.txt", "w") as f:
    f.write(appendixstr)