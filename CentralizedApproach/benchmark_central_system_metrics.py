import os

from sklearn.preprocessing import StandardScaler

from utils.data_utils import load_data, create_X_y_from_gen_df, preprocess_data
from utils.models import get_model
from sklearn.model_selection import StratifiedKFold
import wandb
import tensorflow as tf
from utils.data_utils import preprocess
from utils.config import configs
import argparse

parser = argparse.ArgumentParser(
        prog="benchmark_central_system_metrics.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
parser.add_argument(
    "--run_repeat",type=int, help="num of repeat",default=1
)
parser.add_argument(
    "--num_nodes",type=int,help="num of nodes",default=configs["num_nodes"]
)
# print help if no argument is specified
args = parser.parse_args()

#create train test data
data_path = args.data_path
data_name = data_path.split("/")[2].split(".")[0]
modelname = data_path.split("/")[-1].split(".")[0]
df = load_data(data_path)
df = preprocess_data(df)
X, Y= create_X_y_from_gen_df(df,label=configs["label"])
kfold = StratifiedKFold(n_splits=configs["n_splits"],shuffle=True,random_state=args.run_repeat)

num_nodes = args.num_nodes
dropout_rate = args.dropout_rate
l1_v = args.l1_v
for count,(train,test) in enumerate(kfold.split(X,Y)):
    if count != 0:
        continue
    wandb.init(project=f"benchmark-central_{data_name}_system_metrics", config=configs, job_type='train',group=f"nodes_{num_nodes}_dropout_{dropout_rate}_l1_{l1_v}",name=f"repeat_{args.run_repeat}")
    X_train = X.iloc[train]
    #X_test =X.iloc[test]
    if configs["scale"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        #X_test = scaler.transform(X_test)

    client_dataset = preprocess(tf.data.Dataset.from_tensor_slices((X_train, Y[train])))



    model = get_model(input_dim=X.iloc[train].shape[1], num_nodes=num_nodes,dropout_rate=dropout_rate, l1_v=l1_v, l2_v=configs["l2_v"])
    model.compile(optimizer=configs["optimizer"],
                  loss=configs["loss"],
                  metrics=configs["metrics"])

    begin = tf.timestamp()
    model.fit(client_dataset)
    end = tf.timestamp()
    wandb.log({"training_time": tf.get_static_value(end - begin)})
    #
