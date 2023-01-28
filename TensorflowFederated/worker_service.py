from absl import app
import tensorflow as tf
import tensorflow_federated as tff
from data_loading import GenDataBackend
from absl import flags

# import cygrpc
FLAGS = flags.FLAGS
_PORT = 8000
_GRPC_OPTIONS = [
    ('grpc.max_receive_message_length', 25586421),
    ('grpc.max_send_message_length', 25586421),
    ("grpc.max_metadata_size", 25586421)]
# Number of worker threads in thread pool.
_THREADS = 10
flags.DEFINE_integer("port", 8040, "Sets port of workerservice")




def main(argv) -> None:
    port = FLAGS.port

    def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
        return tff.framework.DataExecutor(
            tff.framework.EagerTFExecutor(device),
            data_backend=GenDataBackend())

    executor_factory = tff.framework.local_executor_factory(
        default_num_clients=1,
        # Max fanout in the hierarchy of local executors
        max_fanout=100,
        leaf_executor_fn=ex_fn)

    print(_GRPC_OPTIONS)
    tff.simulation.run_server(executor_factory, _THREADS, port, None,
                              _GRPC_OPTIONS)


if __name__ == '__main__':
    app.run(main)
