import collections

import grpc
import tensorflow as tf
import tensorflow_federated as tff

from data_loading import FederatedData
from utils.models import get_seq_nn_model

input_spec = collections.OrderedDict([
    ('x', tf.TensorSpec(shape=(None, 12708), dtype=tf.int64, name=None)),
    ('y', tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None))
])
element_type = tff.types.StructWithPythonType(
    input_spec, container_type=collections.OrderedDict)
dataset_type = tff.types.SequenceType(element_type)

train_data_source = FederatedData(type_spec=dataset_type)
train_data_iterator = train_data_source.iterator()


def model_fn():
    model = get_seq_nn_model(input_dim=12708)
    return tff.learning.from_keras_model(
        model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy()])


trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam())


def train_loop(num_rounds=10, num_clients=10):
    state = trainer.initialize()
    for round in range(1, num_rounds + 1):
        train_data = train_data_iterator.select(num_clients)
        result = trainer.next(state, train_data)
        state = result.state
        train_metrics = result.metrics['client_work']['train']
        with open('readme.txt', 'a+') as f:
            f.writelines('round {:2d}, metrics={}'.format(round, train_metrics))
        print('round {:2d}, metrics={}'.format(round, train_metrics))


ip_address_1 = '0.0.0.0'  # @param {type:"string"} TODO change to ip of node
ip_address_2 = '0.0.0.0'  # @param {type:"string"} TODO change to ip of node
port = 80

channels = [
    grpc.insecure_channel(f'{ip_address_1}:{port}'),
    grpc.insecure_channel(f'{ip_address_2}:{port}')
]

tff.backends.native.set_remote_python_execution_context(channels)

train_loop()
