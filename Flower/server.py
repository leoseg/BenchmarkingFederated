import flwr as fl
from Flower.customized_flw_modules.app import start_server
strat = fl.server.strategy.FedAvg(min_fit_clients =1,min_available_clients=1
                                  ,min_evaluate_clients=1)
# Start Flower server
start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy= strat
)