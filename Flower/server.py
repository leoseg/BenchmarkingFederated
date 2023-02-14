from math import ceil

import flwr as fl
from flwr.server import SimpleClientManager
from flwr.server import start_server
import argparse
from Flower.flwr_utils import evaluate_metrics_aggregation_fn
from config import configs
from Flower.customized_flw_modules.server import Server
parser = argparse.ArgumentParser(
        prog="server.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
parser.add_argument(
    "--num_rounds",type=int,help="number of fl rounds", default=1
)
parser.add_argument(
    "--num_clients",type=int,help="number of clients", default=1
)
parser.add_argument(
    "--data_path", type=str, help="path of data to load",default=configs["data_path"]
)
parser.add_argument(
    "--run_repeat",type=int,help="number of run with same config",default=1
)
parser.add_argument(
    "--system_metrics",type=bool,help="flag for system metrics",default=False
)
parser.add_argument(
    "--unweighted_percentage",type=float,help="flag that show that data is that much unweighted",default=-1.0
)
# print help if no argument is specified
args = parser.parse_args()
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "local_epochs": ceil(configs["epochs"]/args.num_rounds)
    }
    return config
if args.system_metrics:
    num_clients = 1
else:
    num_clients = args.num_clients
print (f"Number of clients for server are {num_clients}")
strat = fl.server.strategy.FedAvg(min_fit_clients =num_clients,min_available_clients=num_clients
                                  ,min_evaluate_clients=num_clients,on_fit_config_fn= fit_config,evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
# Start Flower server
print("Starting flowerserver with args:\n")
print(args)
start_server(
    server_address="0.0.0.0:8020",
    server=Server(
        data_path=args.data_path,
        num_clients=args.num_clients,
        client_manager=SimpleClientManager(),
        strategy=strat,
        run_repeat=args.run_repeat,
        system_metrics=args.system_metrics,
        unweighted=args.unweighted_percentage
    ),
    config=fl.server.ServerConfig(num_rounds=args.num_rounds)
)