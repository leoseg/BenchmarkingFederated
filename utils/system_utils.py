from functools import wraps

import pandas as pd
import seaborn as sns
import wandb
import grpc
from config import configs


def get_time_logs(filepath:str,erase=False):
    """
    Reads timestamps from text produced by fl client and returns dict
    :param filepath: path to log file
    :param erase: if true deletes entries afterwards
    :return: dict with client time entry
    """
    file = open(filepath, 'r')
    lines = file.readlines()
    logs = {}
    for count,line in enumerate(lines):
        if line == "Client training time\n":
            logs["client_time"] =float(lines[count+1][:-1])
    if erase:
        open(filepath, 'w').close()
    return logs


def read_system_logs(log_path:str,project_name,group_name,run_name,memory_type):
    """
    Reads data logged from psutil and log to wandb
    :param log_path: path to log file
    :param project_name: wandb project
    :param group_name: wandb group name
    :param run_name: wandb run name
    :param memory_type: type of memory logged (central,client,server)
    :return:
    """
    wandb.init(project=project_name,
               group=group_name, name=run_name,job_type="train",config=configs)
    with open(log_path) as file:
        lines = file.readlines()
        for count,line in enumerate(lines):
            if count == 0:
                continue
            wandb.log({f"memory_{memory_type}": float(line.split()[2])},step=count-1)
            wandb.log({f"cpu_{memory_type}":float(line.split()[1])},step=count-1)
        wandb.log({"total_duration": float(lines[-1].split()[0])})
    total_sum = 0
    log_path_network = log_path.split(".")[0] + "_network.txt"
    with open(log_path_network, "r") as file:
        for line in file:
            entries = line.split()
            if len(entries) >= 40:
                try:
                    entry_40 = float(entries[39])
                    total_sum += entry_40
                except ValueError:
                    pass
    wandb.log({"network_used":total_sum*0.000001})

def draw_group_plot_of_df(list_dfs,time_filter:float=None):
    df = pd.concat(list_dfs,ignore_index=True)
    if time_filter:
        df = df[df["times"]< time_filter]
    plot = sns.relplot(data=df, x="times", y="memory", kind="line",hue="num_clients")
    return plot




import grpc
from grpc._server import _Context

class ByteCountInterceptor(grpc.ServerInterceptor):

    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0

    def intercept_service(self, continuation, handler_call_details):
        print(handler_call_details)

        return continuation(handler_call_details)



