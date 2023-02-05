import argparse
import os
import flwr as fl
from utils.models import get_seq_nn_model
from utils.data_utils import load_gen_data_as_train_test_dataset_balanced,preprocess
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
    "--data_path", type=str, help="path of data to load",default="../DataGenExpression/Dataset1.csv"
)
parser.add_argument(
    "--run_repeat",type=int,help="number of run with same config",default=1
)
parser.add_argument(
    "--system_metrics",type=bool,help="flag for system metrics",default=True
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
if args.client_index:
    rows_to_keep = partitions_list[args.client_index]
else:
    rows_to_keep = None
# Load model and data
train_ds,test_ds = load_gen_data_as_train_test_dataset_balanced(data_path=datapath,rows_to_keep=rows_to_keep,kfold_num=args.random_state,random_state=args.run_repeat)
model = get_seq_nn_model(12708, configs["num_nodes"],configs["dropout_rate"], configs["l1_v"], configs["l2_v"])
model.compile(configs["optimizer"], configs["loss"], metrics=configs["metrics"])


# Define Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        preprocessed_ds = preprocess(train_ds,epochs=config["local_epochs"])
        begin = tf.timestamp()
        model.fit(preprocessed_ds)
        end = tf.timestamp()
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
