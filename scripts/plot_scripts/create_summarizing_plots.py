from utils.plotting import plot_swarmplots, plot_heatmap, \
    create_loss_line_plot, calc_scale
from data_uploading_utils import create_loss_df, group_scenarios, transform_to_df, create_dfs_for_fl_metric
from config import configs
from utils.db_utils import MongoDBHandler
import pandas as pd


mongodb = MongoDBHandler()
for mode in ["unweighted","balanced","system"]:
    plot_path = configs.get("plot_path") + mode + "/"
    unweighted = False
    match mode:
        case "unweighted":
            if configs.get("usecase") == 1 or configs.get("usecase") == 2:
                metric1 = "binary_accuracy_global"
            else:
                metric1 = "sparse_categorical_accuracy_global"
            metric2 = "auc_global"
            metric1_name = "Accuracy"
            metric2_name = "AUC"
            unweighted = True
        case "balanced":
            if configs.get("usecase") == 1 or configs.get("usecase") == 2:
                metric1 = "binary_accuracy"
            else:
                metric1 = "sparse_categorical_accuracy"
            metric2 = "auc"
            metric1_name = "Accuracy"
            metric2_name = "AUC"
        case "system":
            metric1 = "total_memory"
            metric1_name = "Memory usage"
            metric2 = "round_time"
            metric2_name = "Training time"
    for metric,metric_name in zip([metric1,metric2],[metric1_name,metric2_name]):
        scenario_metrics = mongodb.get_data_by_name(f"scenario_metrics_{configs.get('usecase')}_{mode}",True)
        df = create_dfs_for_fl_metric(rounds_metrics=scenario_metrics,metric_name=metric)
        if mode != "unweighted":
            central= mongodb.get_data_by_name(f"central_metrics_{configs.get('usecase')}_{mode}")
            if mode == "balanced":
                central = { key.split("eval_",1)[1] : central[key] for key in central.keys()}
            if metric == "round_time":
                central_metric = "training_time"
            elif metric == "total_memory":
                central_metric = "total_memory_central"
            else:
                central_metric = metric
            central_df = transform_to_df(central,metric_name=central_metric,framework="central",group="central",round_configuration="central")
            df = pd.concat([central_df, df])
        #scale =  calc_scale(df,mode)
        df_pivot = df.pivot_table(index="group", columns="round configuration", values="metric", aggfunc="mean")
        scale = [df_pivot.min().min(), df_pivot.max().max()]
        plot_heatmap(df,"tff",standard_deviation=False,unweighted=unweighted,metric_name=metric_name,data_path=plot_path,scale=scale)
        plot_heatmap(df,"flwr",unweighted=unweighted,scale=scale,metric_name=metric_name,data_path=plot_path)
        df_pivot = df.pivot_table(index="group", columns="round configuration", values="metric", aggfunc="std")
        scale = [df_pivot.min().min(), df_pivot.max().max()]
        plot_heatmap(df,"tff",standard_deviation=True,unweighted=unweighted,metric_name=metric_name,data_path=plot_path,scale=scale)
        plot_heatmap(df,"flwr",standard_deviation=True,unweighted=unweighted,scale=scale,metric_name=metric_name,data_path=plot_path)
        if mode == "unweighted":
            configuration_name = "Percentage of chosen class"
        else:
            configuration_name = "Number of clients"
        for plot_type in ["bar"] :
            plot_swarmplots(df,metric_name,configuration_name,data_path=plot_path,plot_type=plot_type,scale=None)
    # if mode in ["balanced"]:
    # loss_metrics = mongodb.get_data_by_name(f"Loss_data_usecase_{configs.get('usecase')}_{mode}")
    # loss_df = create_loss_df(loss_metrics)
    # create_loss_line_plot(loss_df,plot_path=plot_path)
