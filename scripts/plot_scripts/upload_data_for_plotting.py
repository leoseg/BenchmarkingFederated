import os

from utils.plotting import  get_stats_for_usecase,get_central_metrics,get_loss_stats, create_loss_df,create_loss_line_plot
from config import configs
from utils.db_utils import MongoDBHandler

mongodb = MongoDBHandler()
if configs.get("usecase") == 1:
    version = "unbalanced_with_global_evaluation"

else:
    version = "essential_seeds_42"

for mode in ["balanced","system"]:
    central_loss = {}
    rounds =None
    if mode == "unweighted":
            groups = configs.get("unweighted_groups")
            central = None
            if not configs.get("usecase") == 1:
                version="unbalanced_with_global_evaluation_1804"
    else:
        groups = configs.get("groups")
        central = get_central_metrics(mode=mode, metric_names=[ "eval_"+element.name  for element in configs.get("metrics")])
        if mode != "system":
            central_loss= get_loss_stats(groups=[f"usecase_{configs.get('usecase')}"],mode="central",version="loss_tracking_central")
    if mode == "extreme":
        rounds = [30,100]
        central = None
        version = "extreme"
    if mode != "system":
        loss_metrics = get_loss_stats(groups=groups,mode=mode,version=version)
        loss_metrics.update(central_loss)
        mongodb.update_benchmark_data(data=loss_metrics, name=f"Loss_data_usecase_{configs.get('usecase')}_{mode}")
    scenario_metrics = get_stats_for_usecase(groups,mode=mode,version=version,rounds=rounds)
    mongodb.update_benchmark_data(data=scenario_metrics,name=f"scenario_metrics_{configs.get('usecase')}_{mode}")
    mongodb.update_benchmark_data(data=central,name=f"central_metrics_{configs.get('usecase')}_{mode}")
    data = mongodb.get_data_by_name(f"Loss_data_usecase_{configs.get('usecase')}_{mode}")




