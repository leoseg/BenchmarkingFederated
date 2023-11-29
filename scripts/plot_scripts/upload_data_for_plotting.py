from datapostprocessing.data_uploading_utils import (
    get_loss_stats,
    get_stats_for_usecase,
    get_central_metrics,
)

from datapostprocessing.db_utils import MongoDBHandler
from config import configs

mongodb = MongoDBHandler()
usecase = configs.get("usecase")
if usecase == 1:
    version = "unbalanced_with_global_evaluation"

else:
    version = "unbalanced_with_global_evaluation_1804"

for mode in ["dp"]:
    central_loss = {}
    rounds = None
    if mode == "unweighted":
        groups = configs.get("unweighted_groups")
        central = None
        if not configs.get("usecase") == 1:
            version = "unbalanced_with_global_evaluation_1804"
    elif mode == "dp":
        groups = configs.get("dp_groups")
        version = "dp_noises"
        central = None
    elif mode == "loss":
        central_loss = get_loss_stats(
            groups=[f"usecase_{usecase}"],
            mode="central",
            version="loss_tracking_central",
        )
        groups = configs.get("groups")
        loss_metrics = get_loss_stats(groups=groups, mode=mode, version=version)
        loss_metrics.update(central_loss)
        mongodb.update_benchmark_data(
            data=loss_metrics, name=f"Loss_data_usecase_{configs.get('usecase')}_{mode}"
        )
        continue
    else:
        groups = configs.get("groups")
        central = get_central_metrics(
            mode=mode,
            metric_names=["eval_" + element.name for element in configs.get("metrics")],
        )
    scenario_metrics = get_stats_for_usecase(
        groups, mode=mode, version=version, rounds=rounds
    )
    mongodb.update_benchmark_data(
        data=scenario_metrics,
        name=f"scenario_metrics_{usecase}_{mode}",
    )
    if mode in ["balanced", "system"]:
        mongodb.update_benchmark_data(
            data=central, name=f"central_metrics_{usecase}_{mode}"
        )
