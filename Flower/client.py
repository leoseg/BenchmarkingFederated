import argparse
import os
import flwr as fl
import numpy as np
from keras.utils import set_random_seed

from evaluation_utils import evaluate_model, load_test_data_for_evaluation
from utils.models import get_model
from utils.data_utils import df_train_test_dataset, preprocess, load_data, preprocess_data, log_df_info
import tensorflow as tf
from utils.config import configs
from utils.config import flw_time_logging_directory
import pickle

parser = argparse.ArgumentParser(
        prog="client.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

parser.add_argument(
    "--client_index", type=int, help="index for client to load data partition",default=None
)

parser.add_argument(
    "--data_path", type=str, help="path of data to load",default=configs.get("data_path")
)
parser.add_argument(
    "--run_repeat",type=int,help="number of run with same config",default=1
)
parser.add_argument(
    "--system_metrics",type=bool,help="flag for system metrics",default=False
)
parser.add_argument(
    "--random_state",type=int,help="flag for setting the random state for train test", default=1
)

# print help if no argument is specified
args = parser.parse_args()
with open("partitions_list","rb") as file:
    partitions_list = pickle.load(file)
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
datapath = args.data_path
# If client index flag is None reads whole dataset
if (args.client_index or args.client_index == 0):
    rows_to_keep = partitions_list[args.client_index]
else:
    rows_to_keep = None
# Load and preprocess data
df = load_data(datapath,rows_to_keep)
log_df_info(df, configs["label"])
df = preprocess_data(df)
train_ds,test_ds = df_train_test_dataset(df, kfold_num=args.random_state, random_state=args.run_repeat,label=configs.get("label"),scale=configs.get("scale"))
print("Loading data backend dataset of client has num of examples",train_ds.cardinality())
# Loads and compile model
model = get_model(input_dim=configs.get("input_dim"), num_nodes= configs.get("num_nodes"), dropout_rate=configs.get("dropout_rate"), l1_v= configs.get("l1_v"), l2_v=configs.get("l2_v"))
model.compile(configs.get("optimizer"), configs.get("loss"), metrics=configs.get("metrics"))

# Define Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        tf.print(f"Materializing data for client {args.client_index}"
                 f"Train dataset has size {train_ds.cardinality()}",
                 f"Test dataset has size {test_ds.cardinality()}")
        #for i in range(0,5):
        tf.print(f"train dataset entry {0} from client {args.client_index} is {list(train_ds.as_numpy_iterator())[0]}")
        tf.print(f"test dataset entry {0} from client {args.client_index} is {list(test_ds.as_numpy_iterator())[0]}")
        model.set_weights(parameters)
        preprocessed_ds = preprocess(train_ds,epochs=config["local_epochs"],seed = config["server_round"])
        print(f"epochs are {config['local_epochs']}")
        tf.print(
            f"preprocessed dataset entry {0} from client {args.client_index} is {list(preprocessed_ds.as_numpy_iterator())[0]}")
        begin = tf.timestamp()
        model.fit(preprocessed_ds)
        end = tf.timestamp()
        # If system metrics write client time to file so the server can log it
        if args.system_metrics:
            tf.print("Client training time",output_stream=f"file://{flw_time_logging_directory}")
            tf.print(end-begin,output_stream=f"file://{flw_time_logging_directory}")
        return model.get_weights(), len(list(train_ds)), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        metrics = model.evaluate(test_ds.batch(32),return_dict=True)
        loss = metrics.pop("loss")
        return loss, len(list(test_ds)), metrics



# Start Flower client
print("Starting flowerclient with args:\n")
print(args)
fl.client.start_numpy_client(server_address="127.0.0.1:8020", client=Client())
