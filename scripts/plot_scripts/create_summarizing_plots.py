from utils.plotting import plot_swarmplots,create_dfs_for_fl_metric,transform_to_df, plot_heatmap
from config import configs
from utils.db_utils import MongoDBHandler
import pandas as pd


mongodb = MongoDBHandler()
for mode in ["unweighted"]:
    metric = "binary_accuracy_global"
    scenario_metrics = mongodb.get_data_by_name(f"scenario_metrics_{configs.get('usecase')}_{mode}")
    df = create_dfs_for_fl_metric(rounds_metrics=scenario_metrics,metric_name=metric)
    if mode != "unweighted":
        central= mongodb.get_data_by_name(f"central_metrics_{configs.get('usecase')}_{mode}")
        central = { key.split("eval_",1)[1] : central[key] for key in central.keys()}
        central_df = transform_to_df(central,metric_name=metric,framework="central",group="central",round_configuration="central")
        df = pd.concat([central_df, df])
    scale = plot_heatmap(df,"tff",standard_deviation=False,unweighted=True)
    plot_heatmap(df,"flwr",unweighted=True,scale=scale)
    scale = plot_heatmap(df,"tff",standard_deviation=True,unweighted=True)
    plot_heatmap(df,"flwr",standard_deviation=True,unweighted=True,scale=scale)
    plot_swarmplots(df,metric,"Number of clients")
# for mode in ["balanced"]:
#     metric = "prauc"
#     scenario_metrics = mongodb.get_data_by_name(f"scenario_metrics_{configs.get('usecase')}_{mode}")
#     df = create_dfs_for_fl_metric(rounds_metrics=scenario_metrics, metric_name=metric)
#     if mode != "unweighted":
#         central = mongodb.get_data_by_name(f"central_metrics_{configs.get('usecase')}_{mode}")
#         central_df = transform_to_df(central, metric_name=metric, framework="central", group="central",
#                                      round_configuration="central")
#         df = pd.concat([central_df, df])
#     plot_swarmplots(df, metric, "Percentage of chosen class")
    # metric = "auc"
    # scenario_metrics = mongodb.get_data_by_name(f"scenario_metrics_{configs.get('usecase')}_{mode}")
    # df = create_summarize_dataframe_from_metrics(data=scenario_metrics,metric_name=metric,rounds=[1,3,5,10])
    # round_headers = [ "Round 1 TFF", "Round 1 Flwr","Round 10 TFF", "Round 10 Flwr"]
    # central = mongodb.get_data_by_name(f"central_metrics_{configs.get('usecase')}_{mode}")
    # central_df = transform_central_metric_to_df(central,metric_name=metric)
    #
    # headers = ["Number of clients"] + round_headers
    # transform_df_to_latex(df,headers=headers,rows_to_include=["round 1 tff","round 1 flwr","round 10 tff","round 10 flwr"],central_df=central_df)

