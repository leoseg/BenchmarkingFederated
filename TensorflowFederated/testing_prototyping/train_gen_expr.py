import collections
from tff_config import *
import grpc
import tensorflow as tf
import tensorflow_federated as tff
from customized_tff_modules.fed_avg_with_time import build_weighted_fed_avg
from data_loading import FederatedData
from utils.models import get_seq_nn_model
import wandb
from utils.config import configs
from keras.metrics import BinaryAccuracy

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
    model = get_seq_nn_model(12708, configs.get("num_nodes"),configs.get("dropout_rate"), configs.get("l1_v"), configs.get("l2_v"))
    return tff.learning.from_keras_model(
        model,
        input_spec=element_spec,
        loss=configs.get("loss"),
        metrics=[BinaryAccuracy()])


trainer = build_weighted_fed_avg(
    model_fn,
    use_experimental_simulation_loop=True,
    client_optimizer_fn=lambda: configs.get("optimizer"),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    model_aggregator=tff.learning.robust_aggregator(zeroing=False, clipping=False, debug_measurements_fn=tff.learning.add_debug_measurements),
)


def train_loop(num_rounds=1, num_clients=1):
    begin = tf.timestamp()
    state = trainer.initialize()
    end = tf.timestamp()

    tf.print("-------------------------------------------",output_stream="file://worker_service_logging.out")
    with open('../timelogs/worker_service_logging.out', 'a+') as f:
        f.writelines(
            f"\nNew run with config num_rounds:{NUM_ROUNDS},num_clients:{num_clients},epochs:{EPOCHS},batch:{BATCH}"
        )

    round_data_uris = [f'uri://{i}' for i in range(num_clients)]
    round_train_data = tff.framework.CreateDataDescriptor(
        arg_uris=round_data_uris, arg_type=dataset_type)
    tf.print(f"\nInitializationtime is {end - begin}",output_stream="file://worker_service_logging.out")
    logdir="testlog"
    for round in range(1, num_rounds + 1):
        begin = tf.timestamp()
        result = trainer.next(state, round_train_data)
        end = tf.timestamp()
        round_time = end -begin
        tf.print(f"Round  {round} time is {round_time}",output_stream="file://worker_service_logging.out")
        state = result.state
        train_metrics = result.metrics['client_work']['train']
        with open('../timelogs/worker_service_logging.out', 'a+') as f:
            f.writelines('Metrics={}'.format(train_metrics))
    tf.print("\n-------------------------------------------", output_stream="file://worker_service_logging.out")

ip_address= '0.0.0.0'  # @param {type:"string"} TODO change to ip of node
port1 = 8040
port2 = 8050
port3 = 8060

channels = [
    grpc.insecure_channel(f'{ip_address}:{port1}',options=[ ('grpc.max_send_message_length', 25586421),
        ('grpc.max_receive_message_length',25586421), ("grpc.max_metadata_size",25586421)]),
 grpc.insecure_channel(f'{ip_address}:{port2}',options=[ ('grpc.max_send_message_length', 25586421),
          ('grpc.max_receive_message_length',25586421),("grpc.max_metadata_size",25586421)]),
 grpc.insecure_channel(f'{ip_address}:{port3}',options=[ ('grpc.max_send_message_length', 25586421),
          ('grpc.max_receive_message_length',25586421),("grpc.max_metadata_size",25586421)])
]

tff.backends.native.set_remote_python_execution_context([channels[1]])


train_loop(NUM_ROUNDS,NUM_CLIENTS)
