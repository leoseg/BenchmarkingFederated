from math import ceil
from typing import Optional, Tuple, Dict

import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server import SimpleClientManager
from flwr.server import start_server
import argparse
from Flower.flwr_utils import evaluate_metrics_aggregation_fn
from config import configs
from Flower.customized_flw_modules.server import Server
from evaluation_utils import load_test_data_for_evaluation, evaluate_model
from models import get_model

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
    "--data_path", type=str, help="path of data to load",default=configs.get("data_path")
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
parser.add_argument(
    "--network_metrics",type=bool,help="flag for network metrics",default=False
)
parser.add_argument(
    "--noise",type=float,help="dp_noise",default=0.0
)
# print help if no argument is specified
args = parser.parse_args()
def fit_config(server_round: int):
    """Return training configuration dict for each round. Sets local epochs of each client"""
    config = {
        "local_epochs": ceil(configs.get("epochs")/args.num_rounds),
        "server_round": server_round
    }
    return config

num_clients = args.num_clients
model = get_model(input_dim=configs.get("input_dim"), num_nodes=configs.get("num_nodes"),
                  dropout_rate=configs.get("dropout_rate"), l1_v=configs.get("l1_v"), l2_v=configs.get("l2_v"))
model.compile(configs.get("optimizer"), configs.get("loss"), metrics=configs.get("metrics"))

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself


    # The `evaluate` function will be called after every round
    X_test, y_test = load_test_data_for_evaluation(args.run_repeat)
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        new_parameters = parameters
        model.set_weights(new_parameters)
        metrics = model.evaluate(X_test, y_test, verbose = 0,return_dict=True)
        loss = metrics.pop("loss")
        return loss,metrics

    return evaluate


print (f"Number of clients for server are {num_clients}")
if args.unweighted_percentage >= 0.0:
    strat = fl.server.strategy.FedAvg(min_fit_clients =num_clients,min_available_clients=num_clients
                                      ,min_evaluate_clients=num_clients,on_fit_config_fn= fit_config,
                                      evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                      evaluate_fn=get_evaluate_fn(model=model))
else:
    strat = fl.server.strategy.FedAvg(min_fit_clients=num_clients, min_available_clients=num_clients
                                      , min_evaluate_clients=num_clients, on_fit_config_fn=fit_config,
                                      evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                )
# Start Flower server
print("Starting flowerserver with args:\n")
print(args)
start_server(
    server_address="0.0.0.0:8150",
    server=Server(
        data_path=args.data_path,
        num_clients=args.num_clients,
        client_manager=SimpleClientManager(),
        strategy=strat,
        run_repeat=args.run_repeat,
        system_metrics=args.system_metrics,
        unweighted=args.unweighted_percentage,
        network_metrics=args.network_metrics,
        noise=args.noise
    ),
    config=fl.server.ServerConfig(num_rounds=args.num_rounds)
)