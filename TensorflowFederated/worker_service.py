from absl import app
import tensorflow as tf
import tensorflow_federated as tff
from collections.abc import Sequence
from data_loading import GenDataBackend

_PORT = 8000
_GRPC_OPTIONS = [('grpc.max_message_length', 20 * 1024 * 1024),
                 ('grpc.max_receive_message_length', 20 * 1024 * 1024)]
# Number of worker threads in thread pool.
_THREADS = 10


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
        return tff.framework.DataExecutor(
            tff.framework.EagerTFExecutor(device),
            data_backend=GenDataBackend())

    executor_factory = tff.framework.local_executor_factory(
        default_num_clients=1,
        # Max fanout in the hierarchy of local executors
        max_fanout=100,
        leaf_executor_fn=ex_fn)

    tff.simulation.run_server(executor_factory, _THREADS, _PORT, None,
                              _GRPC_OPTIONS)


if __name__ == '__main__':
    app.run(main)
