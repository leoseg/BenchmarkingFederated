from utils.data_utils import load_gen_data, create_X_y
from utils.models import get_seq_nn_model
from sklearn.model_selection import StratifiedKFold
import wandb
import tensorflow as tf
from utils.data_utils import preprocess
from utils.config import configs
import argparse

parser = argparse.ArgumentParser(
        prog="benchmark_central.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

parser.add_argument(
    "--num_nodes", type=int, help="num of nodes for each layer",default=configs["num_nodes"]
)

parser.add_argument(
    "--dropout_rate", type=float, help="dropout rate",default=configs["dropout_rate"]
)
parser.add_argument(
    "--l1_v",type=float,help="l1 kernel regularizer",default=configs["l1_v"]
)
parser.add_argument(
    "--data_path", type=str, help="path of data to load",default=configs["data_path"]
)
# print help if no argument is specified
args = parser.parse_args()

#create train test data
data_path = args.data_path
data_name = data_path.split("/")[2].split(".")[0]
modelname = data_path.split("/")[-1].split(".")[0]
df = load_gen_data(data_path)
X, Y= create_X_y(df)
kfold = StratifiedKFold(n_splits=configs["n_splits"],shuffle=True,random_state=0)

num_nodes = args.num_nodes
dropout_rate = args.dropout_rate
l1_v = args.l1_v
for count,(train,test) in enumerate(kfold.split(X,Y)):
    wandb.init(project=f"benchmark-central_{data_name}_with_tf_dataset_and_time", config=configs, job_type='train',group=f"nodes_{num_nodes}_dropout_{dropout_rate}_l1_{l1_v}",name=f"k_fold_{count}")

    client_dataset = preprocess(tf.data.Dataset.from_tensor_slices((X.iloc[train], Y[train])))



    model = get_seq_nn_model(X.iloc[train].shape[1], num_nodes,dropout_rate, l1_v, configs["l2_v"])
    model.compile(optimizer=configs["optimizer"],
                  loss=configs["loss"],
                  metrics=configs["metrics"])

    begin = tf.timestamp()
    model.fit(client_dataset)
    end = tf.timestamp()
    wandb.log({"training_time":end-begin})
    #evaluate utils
    score = model.evaluate(X.iloc[test], Y[test], verbose = 0,return_dict=True)
    for key,value in score.items():
        wandb.log({f"eval_{key}": value})
    wandb.finish()
