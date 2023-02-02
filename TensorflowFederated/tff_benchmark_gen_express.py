import collections
from tff_config import *
import grpc
import tensorflow as tf
import tensorflow_federated as tff
from customized_tff_modules.fed_avg_with_time import build_weighted_fed_avg
from data_loading import FederatedData
from utils.time_logging import get_time_logs
from utils.models import get_seq_nn_model
import wandb
from utils.config import configs
from utils.config import tff_time_logging_directory
import argparse
parser = argparse.ArgumentParser(
        prog="train_gen_expr.py",
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

element_spec = (
    tf.TensorSpec(shape=(None, 12708), dtype=tf.float64, name=None),
tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None)
)

element_type = tff.types.StructWithPythonType(
    element_spec,
    container_type=collections.OrderedDict)
dataset_type = tff.types.SequenceType(element_type)

train_data_source = FederatedData(type_spec=dataset_type)
train_data_iterator = train_data_source.iterator()


def model_fn():
    model = get_seq_nn_model(12708, configs["num_nodes"],configs["dropout_rate"], configs["l1_v"], configs["l2_v"])
    return tff.learning.from_keras_model(
        model,
        input_spec=element_spec,
        loss=configs["loss"],
        metrics=[configs["metrics"]])


trainer = build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: configs["optimizer"],
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    model_aggregator=tff.learning.robust_aggregator(zeroing=False, clipping=False, debug_measurements_fn=tff.learning.add_debug_measurements))
evaluation_process = tff.learning.algorithms.build_fed_eval(model_fn=model_fn)
data_name = args.data_path.split("/")[-1].split(".")[0]
if args.system_metrics:
    metrics_type = "system"
else:
    metrics_type = "model"

wandb.init(project=f"benchmark_rounds_{args.num_rounds}_{data_name}_{metrics_type}_metrics", group=f"tff_{args.num_clients}", name=f"run_{args.run_repeat}")

def train_loop(num_rounds=1, num_clients=1):
    evaluation_state = evaluation_process.initialize()
    state = trainer.initialize()
    round_data_uris = [f'uri://{i}' for i in range(num_clients)]
    round_train_data = tff.framework.CreateDataDescriptor(
        arg_uris=round_data_uris, arg_type=dataset_type)
    eval_data_uris = [f'e{i}' for i in range(num_clients)]
    eval_data = tff.framework.CreateDataDescriptor(
        arg_uris=eval_data_uris, arg_type=dataset_type)
    for round in range(1, num_rounds + 1):
        begin = tf.timestamp()
        result = trainer.next(state, round_train_data)
        end = tf.timestamp()
        state = result.state
        model_weights = trainer.get_model_weights(state)
        evaluation_state = evaluation_process.set_model_weights(evaluation_state, model_weights)
        evaluation_output = evaluation_process.next(evaluation_state, eval_data)
        if not args.system_metrics:
            wandb.log(evaluation_output.metrics["client_work"]["current_round_metrics"])

        if args.system_metrics:
            wandb.log({"round_time",end-begin})
            wandb.log(get_time_logs(tff_time_logging_directory,True))




ip_address= '0.0.0.0'
ports = []
port_num = 8000
channels =[]
for i in range(1,args.num_clients+1):
    channels.append(grpc.insecure_channel(f'{ip_address}:{port_num+i}',options=[ ('grpc.max_send_message_length', 25586421),
        ('grpc.max_receive_message_length',25586421), ("grpc.max_metadata_size",25586421)]),)


tff.backends.native.set_remote_python_execution_context(channels)
train_loop(args.num_rounds,args.num_clients)
