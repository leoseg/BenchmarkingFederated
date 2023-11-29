import collections
import concurrent.futures
import pickle

import time
from dp.dp_query import LocalGaussianSumQuery
import grpc
import tensorflow as tf
import tensorflow_federated as tff

from customized_tff_modules.fed_avg_with_time import build_unweighted_fed_avg
from utils.system_utils import get_time_logs
from utils.models import get_model
import wandb

from utils.config import configs
from utils.config import tff_time_logging_directory
import argparse
from keras.metrics import (
    AUC,
    BinaryAccuracy,
    Recall,
    Precision,
    SparseCategoricalAccuracy,
)
from metrics import AUC as SparseAUC
import os
from keras.optimizers import Adam

parser = argparse.ArgumentParser(
    prog="tff_dp.py",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--num_clients", type=int, help="number of fl rounds", default=3)
parser.add_argument("--noise", type=float, help="dp_noise", default=0.0)
parser.add_argument(
    "--data_path",
    type=str,
    help="path of data to load",
    default=configs.get("data_path"),
)
parser.add_argument(
    "--run_repeat", type=int, help="number of run with same config", default=0
)
parser.add_argument(
    "--system_metrics", type=bool, help="flag for system metrics", default=False
)
parser.add_argument("--num_rounds", type=int, help="number of fl rounds", default=1)
# print help if no argument is specified
args = parser.parse_args()
noise = float(args.noise)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
element_spec = (
    tf.TensorSpec(shape=(None, configs["input_dim"]), dtype=tf.float64, name=None),
    tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None),
)
print("Command line args are:\n")
print(args)
element_type = tff.types.StructWithPythonType(
    element_spec, container_type=collections.OrderedDict
)
dataset_type = tff.types.SequenceType(element_type)

# train_data_source = FederatedData(type_spec=dataset_type)
# train_data_iterator = train_data_source.iterator()


# Model function to use for FL
def model_fn():
    """
    Creates keras model and wraps it in a tff learning model
    """
    model = get_model(
        input_dim=configs.get("input_dim"),
        num_nodes=configs.get("num_nodes"),
        dropout_rate=configs.get("dropout_rate"),
        l1_v=configs.get("l1_v"),
        l2_v=configs.get("l2_v"),
    )
    # Chooses metrics depending on usecase
    if configs["usecase"] == 3 or configs["usecase"] == 4:
        metrics = [
            SparseCategoricalAccuracy(),
            SparseAUC(name="auc"),
            SparseAUC(curve="PR", name="prauc"),
        ]
    else:
        metrics = [
            BinaryAccuracy(),
            AUC(),
            Precision(),
            Recall(),
            AUC(curve="PR", name="prauc"),
        ]
    return tff.learning.from_keras_model(
        model, input_spec=element_spec, loss=configs.get("loss"), metrics=metrics
    )


# Build federated learning process
# Uses customized classes that measure train time of clients and write that to a file
# aggregator = tff.learning.ddp_secure_aggregator(noise_multiplier=1.0,expected_clients_per_round=1000)
aggregator = tff.learning.robust_aggregator(
    zeroing=False, clipping=False, weighted=False
)
# # print(model_fn().trainable_variables)
# # query = dp_utils.create_compression_sum_query(
# #     1.0,
# #     tfp.DistributedDiscreteGaussianSumQuery(0.5,args.noise),
# # model_fn().trainable_variables)

aggregator = tff.aggregators.DifferentiallyPrivateFactory(
    LocalGaussianSumQuery(configs.get("global_norm"), args.noise), aggregator
)
# optimizer = tfp.DPKerasAdamOptimizer(l2_norm_clip=1.0,noise_multiplier=noise,num_microbatches=1)
optimizer = Adam()
# momentum = 0.9
momentum = 0.0
print(aggregator)
trainer = build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: optimizer,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(
        learning_rate=1.0, momentum=momentum
    ),
    model_aggregator=aggregator,
)
# Build federated evaluation process
evaluation_process = tff.learning.algorithms.build_fed_eval(model_fn=model_fn)
data_name = args.data_path.split("/")[-1].split(".")[0]


if args.system_metrics:
    mode = "system"
    num_clients = 1
else:
    mode = "model"
    num_clients = args.num_clients
project_name = f"dpusecase_{configs['usecase']}_benchmark_rounds_{args.num_rounds}_{data_name}_{mode}_metrics"
group = f"tff_{args.num_clients}_{noise}"
print("Training initialized")
DELAY_SECONDS = 5  # Delay between each retry attempt

while True:
    try:
        wandb.init(
            project=project_name,
            group=group,
            name=f"run_{args.run_repeat}",
            config=configs,
            settings=wandb.Settings(_service_wait=300),
        )
        print("Wandb initialized successfully")
        break
    except ConnectionRefusedError:
        print(f"Connection refused. Retrying in {DELAY_SECONDS} seconds...")
        time.sleep(DELAY_SECONDS)
with open("partitions_list", "rb") as file:
    partitions_list = pickle.load(file)
wandb.log({"partitions_list": partitions_list})
num_examples_per_client = int(len(partitions_list[0]) * 0.8)
# privacy_guarantee = tfp.compute_dp_sgd_privacy(num_examples_per_client,configs.get("batch_size"),noise,configs.get("epochs"),delta=calculate_delta(num_examples_per_client))
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
    eval_data_uris = [f"e{i}" for i in range(num_clients)]
    eval_data = tff.framework.CreateDataDescriptor(
        arg_uris=eval_data_uris, arg_type=dataset_type
    )
    model = get_model(
        input_dim=configs.get("input_dim"),
        num_nodes=configs.get("num_nodes"),
        dropout_rate=configs.get("dropout_rate"),
        l1_v=configs.get("l1_v"),
        l2_v=configs.get("l2_v"),
    )
    model.compile(
        configs.get("optimizer"), configs.get("loss"), metrics=configs.get("metrics")
    )
    # Loop trough rounds
    for round in range(1, num_rounds + 1):
        print(f"Begin round {round}")
        begin = tf.timestamp()
        # Do training round with state before
        round_data_uris = [f"{round}_uri://{i}" for i in range(num_clients)]
        round_train_data = tff.framework.CreateDataDescriptor(
            arg_uris=round_data_uris, arg_type=dataset_type
        )
        result = trainer.next(state, round_train_data)
        end = tf.timestamp()
        # If not system metrics gets weights from averaged model and uses that for evaluation on clients
        # then log to wandb
        state = result.state
        train_metrics = result.metrics
        print(train_metrics)
        if not args.system_metrics:
            model_weights = trainer.get_model_weights(state)
            evaluation_state = evaluation_process.set_model_weights(
                evaluation_state, model_weights
            )
            evaluation_output = evaluation_process.next(evaluation_state, eval_data)
            # wandb.log({"training_loss": train_metrics["loss"]}, step=round)
            wandb.log(
                evaluation_output.metrics["client_work"]["eval"][
                    "current_round_metrics"
                ],
                step=round,
            )

        if args.system_metrics:
            round_time = end - begin
            wandb.log({"round_time": tf.get_static_value(round_time)}, step=round)
            wandb.log(get_time_logs(tff_time_logging_directory, True), step=round)


ip_address = "0.0.0.0"
ports = []
port_num = 8000
channels = []
executor = concurrent.futures.ThreadPoolExecutor()
# Creates channels for each client for communication
for i in range(1, num_clients + 1):
    channels.append(
        grpc.insecure_channel(
            f"{ip_address}:{port_num+i}",
            options=[
                ("grpc.max_send_message_length", 25586421),
                ("grpc.max_receive_message_length", 25586421),
                ("grpc.max_metadata_size", 25586421),
            ],
        ),
    )

# Sets remote execution
tff.backends.native.set_remote_python_execution_context(
    channels, thread_pool_executor=executor
)
# Start FL
train_loop(args.num_rounds, num_clients)
