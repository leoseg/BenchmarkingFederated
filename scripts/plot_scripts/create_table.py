import pandas as pd

from table_utils import transform_df_to_latex, create_summarize_dataframe_from_metrics, transform_central_metric_to_df, transform_df_to_landscape_table
from db_utils import MongoDBHandler

mongodb = MongoDBHandler()
data = mongodb.get_data_by_name("scenario_metrics_1_balanced")
central = mongodb.get_data_by_name("central_metrics_1_balanced")
groups = ["tff_3","flwr_3","tff_5","flwr_5","tff_10","flwr_10"]
df = create_summarize_dataframe_from_metrics(data=data,metric_name="auc",rounds=[1,3,5,10],groups=groups)
central_df = transform_central_metric_to_df(central,metric_name="eval_auc")
headers = ["Number of clients","Round 1 TFF", "Round 1 Flwr","Round 3 TFF", "Round 3 Flwr","Round 5 TFF", "Round 5 Flwr","Round 10 TFF", "Round 10 Flwr", "Group Summary TFF", "Group Summary FLWR"]
latex_str = transform_df_to_landscape_table(df,headers,"Metrics for AUC with different Number of Clients",central_df=central_df)
