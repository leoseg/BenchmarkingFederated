from absl import app
import tensorflow as tf
import tensorflow_federated as tff
from data_loading import DataBackend
from absl import flags
from data_utils import df_train_test_dataset, load_data, preprocess_data, log_df_info
from utils.config import configs
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
FLAGS = flags.FLAGS
_GRPC_OPTIONS = [
    ('grpc.max_receive_message_length', 25586421),
    ('grpc.max_send_message_length', 25586421),
    ("grpc.max_metadata_size", 25586421)]
# Number of worker threads in thread pool.
_THREADS = 1
flags.DEFINE_integer("port", 8050, "Sets port of workerservice")
flags.DEFINE_integer("num_rounds",1,"Defines number of rounds")
flags.DEFINE_integer("client_index",None,"index for client to load data partition")
flags.DEFINE_string("data_path",configs.get("data_path"),"Defines path to data")
flags.DEFINE_integer("run_repeat",1,"number of run with same config")
flags.DEFINE_integer("random_state",1,"random state for train test split")
flags.DEFINE_bool("unweighted",False,"flag for unweighted federated averaging")


def main(argv) -> None:
    port = FLAGS.port
    num_rounds = FLAGS.num_rounds
    epochs = int(configs.get("epochs")/num_rounds)
    data_path  = FLAGS.data_path
    run_repeat = FLAGS.run_repeat
    random_state = FLAGS.random_state
    client_index = FLAGS.client_index
    # If client index flag is None reads whole dataset
    if (client_index or client_index == 0):
        with open("partitions_list", "rb") as file:
            partitions_list = pickle.load(file)
        rows_to_keep = partitions_list[client_index]
    else:
        rows_to_keep = None
    # Loads and preprocesses data
    df = load_data(data_path,rows_to_keep)
    df = preprocess_data(df)
    log_df_info(df, configs["label"])
    train_dataset, test_dataset = df_train_test_dataset(
        df,
        kfold_num=random_state,
        random_state=run_repeat,
        label=configs.get("label"),
        scale=configs.get("scale"),
        unweighted=FLAGS.unweighted
    )
    # Sets executor for local calculations
    def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
        return tff.framework.DataExecutor(
            tff.framework.EagerTFExecutor(device),
            data_backend=DataBackend(
                local_epochs=epochs,
                train_dataset=train_dataset,
                test_dataset=test_dataset))


    executor_factory = tff.framework.local_executor_factory(
        default_num_clients=1,
        # Max fanout in the hierarchy of local executors,
        max_fanout=100,
        leaf_executor_fn=ex_fn)

    print(f"Worker created with port {port}")
    tff.simulation.run_server(executor_factory, _THREADS, port, None,
                              _GRPC_OPTIONS)


if __name__ == '__main__':
    app.run(main)
