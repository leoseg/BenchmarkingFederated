from absl import app
import tensorflow as tf
import tensorflow_federated as tff
from data_loading import GenDataBackend
from absl import flags
from utils.config import configs
# import cygrpc
FLAGS = flags.FLAGS
_GRPC_OPTIONS = [
    ('grpc.max_receive_message_length', 25586421),
    ('grpc.max_send_message_length', 25586421),
    ("grpc.max_metadata_size", 25586421)]
# Number of worker threads in thread pool.
_THREADS = 1
flags.DEFINE_integer("port", 8040, "Sets port of workerservice")
flags.DEFINE_integer("num_rounds",1,"Defines number of rounds")
flags.DEFINE_list("rows_to_keep",[1,2],"Defines rows to keep")
flags.DEFINE_string("data_path","Dataset1.csv","Defines path to data")
flags.DEFINE_integer("run_repeat",1,"number of run with same config")

def main(argv) -> None:
    port = FLAGS.port
    num_rounds = FLAGS.num_rounds
    epochs = configs["epochs"]/num_rounds
    rows_to_keep = FLAGS.rows_to_keep
    data_path  = FLAGS.data_path
    run_repeat = FLAGS.repeat_num

    def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
        return tff.framework.DataExecutor(
            tff.framework.EagerTFExecutor(device),
            data_backend=GenDataBackend(
                local_epochs=epochs,
            rows_to_keep=rows_to_keep,
            data_path=data_path,
            kfold_num=run_repeat))

    executor_factory = tff.framework.local_executor_factory(
        default_num_clients=1,
        # Max fanout in the hierarchy of local executors
        max_fanout=100,
        leaf_executor_fn=ex_fn)

    tff.simulation.run_server(executor_factory, _THREADS, port, None,
                              _GRPC_OPTIONS)


if __name__ == '__main__':
    app.run(main)
