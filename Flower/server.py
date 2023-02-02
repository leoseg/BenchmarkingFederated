import flwr as fl
from flwr.server import SimpleClientManager
from flwr.server import start_server
import argparse
from utils.config import configs
from Flower.customized_flw_modules.server import Server
parser = argparse.ArgumentParser(
        prog="server.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
parser.add_argument(
    "--num_rounds",type=int,help="number of fl rounds"
)
parser.add_argument(
    "--num_clients",type=int,help="number of clients"
)
parser.add_argument(
    "--datapath", type=str, help="path of data to load"
)
parser.add_argument(
    "--run_repeat",type=int,help="number of run with same config"
)
parser.add_argument(
    "--system_metrics",type=bool,help="flag for system metrics"
)
# print help if no argument is specified
args = parser.parse_args()

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "local_epochs": int(configs["epochs"]/args.num_rounds)
    }
    return config

strat = fl.server.strategy.FedAvg(min_fit_clients =1,min_available_clients=1
                                  ,min_evaluate_clients=1,on_fit_config= fit_config)
# Start Flower server
start_server(
    server_address="0.0.0.0:8080",
    server=Server(
        data_path=args.datapath,
        num_clients=args.num_clients,
        client_manager=SimpleClientManager(),
        strategy=strat,
        run_repeat=args.run_repeat,
        system_metrics=args.system_metrics
    ),
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy= strat
)