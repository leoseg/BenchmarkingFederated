import collections
import concurrent.futures
import pickle

from py_grpc_prometheus.prometheus_client_interceptor import PromClientInterceptor
from TensorflowFederated.testing_prototyping.tff_config import *
import grpc
import tensorflow as tf
import tensorflow_federated as tff
from customized_tff_modules.fed_avg_with_time import build_weighted_fed_avg
from utils.system_utils import get_time_logs
from utils.models import get_model
import wandb
from utils.config import configs
from utils.config import tff_time_logging_directory
import argparse
from keras.metrics import AUC,BinaryAccuracy,Recall,Precision, SparseCategoricalAccuracy
from metrics import AUC as SparseAUC
import os
import pandas as pd
parser = argparse.ArgumentParser(
        prog="train_gen_expr.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
parser.add_argument(
    "--num_rounds",type=int,help="number of fl rounds",default=2
)
parser.add_argument(
    "--num_clients",type=int,help="number of clients",default=2
)
parser.add_argument(
    "--data_path", type=str, help="path of data to load",default=configs.get("data_path")
)
parser.add_argument(
    "--run_repeat",type=int,help="number of run with same config",default=0
)
parser.add_argument(
    "--system_metrics",type=bool,help="flag for system metrics",default=False
)
parser.add_argument(
    "--unweighted_percentage",type=float,help="flag that show that data is that much unweighted",default=-1.0
)
# print help if no argument is specified
args = parser.parse_args()
unweighted = args.unweighted_percentage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
element_spec = (
    tf.TensorSpec(shape=(None, configs["input_dim"]), dtype=tf.float64, name=None),
tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None)
)
print("Command line args are:\n")
print(args)
element_type = tff.types.StructWithPythonType(
    element_spec,
    container_type=collections.OrderedDict)
dataset_type = tff.types.SequenceType(element_type)

# train_data_source = FederatedData(type_spec=dataset_type)
# train_data_iterator = train_data_source.iterator()


# Model function to use for FL
def model_fn():
    model = get_model(input_dim=configs.get("input_dim"), num_nodes= configs.get("num_nodes"), dropout_rate=configs.get("dropout_rate"), l1_v= configs.get("l1_v"), l2_v=configs.get("l2_v"))
    # Chooses metrics depending on usecase
    if configs["usecase"] ==3 or configs["usecase"] == 4:
        metrics = [SparseCategoricalAccuracy(),SparseAUC(name="auc"),SparseAUC(curve="PR",name="prauc")]
    else:
        metrics = [BinaryAccuracy(),AUC(),Precision(),Recall(),AUC(curve="PR",name="prauc")]
    return tff.learning.from_keras_model(
        model,
        input_spec=element_spec,
        loss=configs.get("loss"),
        metrics=metrics)

# Build federated learning process
# Uses customized classes that measure train time of clients and write that to a file
trainer = build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: configs.get("optimizer"),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    model_aggregator=tff.learning.robust_aggregator(zeroing=False, clipping=False, debug_measurements_fn=tff.learning.add_debug_measurements))
# Build federated evaluation process
evaluation_process = tff.learning.algorithms.build_fed_eval(model_fn=model_fn)
data_name = args.data_path.split("/")[-1].split(".")[0]
if args.system_metrics == True:
    metrics_type = "system"
    num_clients = 1
else:
    metrics_type = "model"
    num_clients = args.num_clients


project_name = f"benchmark_rounds_{args.num_rounds}_{data_name}_{metrics_type}_metrics"
project_name = f"usecase_{configs['usecase']}_" + project_name
if unweighted >= 0.0:
    project_name = "unweighted" + project_name
    group = f"tff_{args.unweighted_percentage}"

else:
    group = f"tff_{args.num_clients}"
print("Training initialized")
wandb.init(project=project_name, group=group, name=f"run_{args.run_repeat}",config=configs)
with open("partitions_list", "rb") as file:
    partitions_list = pickle.load(file)
wandb.log({"partitions_list": partitions_list})
# If unweighted log number of samples per class per client
if unweighted >= 0.0:
    wandb.log({"class_num_table":pd.read_csv("partitions_dict.csv")})
def train_loop(num_rounds=1, num_clients=1):
    """
    Train loop function for FL
    :param num_rounds: number of rounds FL
    :param num_clients: number of clients for FL
    :return:
    """
    evaluation_state = evaluation_process.initialize()
    state = trainer.initialize()
    print("inital weights are:")
    print(trainer.get_model_weights(state).trainable)
    # round_data_uris = [f'uri://{i}' for i in range(num_clients)]
    # round_train_data = tff.framework.CreateDataDescriptor(
    #     arg_uris=round_data_uris, arg_type=dataset_type)
    eval_data_uris = [f'e{i}' for i in range(num_clients)]
    eval_data = tff.framework.CreateDataDescriptor(
        arg_uris=eval_data_uris, arg_type=dataset_type)
    # Loop trough rounds

    for round in range(1, num_rounds + 1):
        print(f"Begin round {round}")
        begin = tf.timestamp()
        # Do training round with state before
        round_data_uris = [f'{round}_uri://{i}' for i in range(num_clients)]
        round_train_data = tff.framework.CreateDataDescriptor(
            arg_uris=round_data_uris, arg_type=dataset_type)
        result = trainer.next(state, round_train_data)
        end = tf.timestamp()
        # If not system metrics gets weights from averaged model and uses that for evaluation on clients
        # then log to wandb
        state = result.state
        print("weights after round {round} are:")
        print(trainer.get_model_weights(state).trainable)
        if not args.system_metrics:
            model_weights = trainer.get_model_weights(state)
            evaluation_state = evaluation_process.set_model_weights(evaluation_state, model_weights)
            evaluation_output = evaluation_process.next(evaluation_state, eval_data)
            wandb.log(evaluation_output.metrics["client_work"]["eval"]["current_round_metrics"])

        if args.system_metrics:
            round_time = end-begin
            wandb.log({"round_time":tf.get_static_value(round_time)},step=round)
            wandb.log(get_time_logs(tff_time_logging_directory,True),step=round)
            wandb.log()
            #network_usage = interceptor.received + interceptor.sent
            #wandb.log({"network_usage": interceptor._metrics},step=round)
    # weights = trainer.get_model_weights(state)
    # # Save model weights
    # model = get_model(input_dim=configs.get("input_dim"), num_nodes= configs.get("num_nodes"), dropout_rate=configs.get("dropout_rate"), l1_v= configs.get("l1_v"), l2_v=configs.get("l2_v"))
    # model.set_weights(weights.trainable)
    # model.save_weights(f"tff_weights.h5")



ip_address= '0.0.0.0'
ports = []
port_num = 8000
channels =[]
executor = concurrent.futures.ThreadPoolExecutor()
# Creates channels for each client for communication
for i in range(1,num_clients+1):
    channels.append(grpc.insecure_channel(f'{ip_address}:{port_num+i}',options=[ ('grpc.max_send_message_length', 25586421),
        ('grpc.max_receive_message_length',25586421), ("grpc.max_metadata_size",25586421)]),)

# Sets remote execution
#start_http_server(8020)
tff.backends.native.set_remote_python_execution_context(channels,thread_pool_executor=executor)
# Start FL
train_loop(args.num_rounds,num_clients)
