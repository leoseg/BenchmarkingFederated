from utils.data_utils import load_gen_data, create_X_y, load_gen_data_as_train_test_split
from utils.models import get_seq_nn_model
from sklearn.model_selection import StratifiedKFold
import wandb
from wandb.keras import WandbCallback
from utils.config import configs
import argparse
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(
        prog="benchmark_central_model_metrics.py",
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
for epochs in [100]:
    configs["epochs"] = epochs
    data_pathes = ["../DataGenExpression/Alldata.csv"]
    for data_path in data_pathes:
        data_name = data_path.split("/")[2].split(".")[0]
        modelname = data_path.split("/")[-1].split(".")[0]
        df = load_gen_data(data_path)
        X, Y = create_X_y(df,False)
        random_state = 69
        kfold = StratifiedKFold(n_splits=configs["n_splits"], shuffle=True, random_state=random_state)

        num_nodes = args.num_nodes
        dropout_rate = args.dropout_rate
        l1_v = args.l1_v
        group_name= f"no_crossfold_{random_state}_{num_nodes}_dropout_{dropout_rate}_l1_{l1_v}"
        if epochs ==200:
            group_name = "200_epochs_" +group_name
        wandb.init(project=f"choose-best-config-central_{data_name}_gen_expr", config=configs,group=group_name,job_type='train',name=f"no_crossfold")
            # client_dataset = tf.data.Dataset.from_tensor_slices((X.iloc[train], Y[train]))
            # Define WandbCallback for experiment tracking
        wandb_callback = WandbCallback(monitor='val_loss',
                                       log_weights=True,
                                       log_evaluation=True,
                                       save_model=False,
                                       save_weights_only=True)
            # dataset_size = len(list(client_dataset))
        X_train, X_test, y_train, y_test = load_gen_data_as_train_test_split(data_path)
        model = get_seq_nn_model(X_train.shape[1], configs["num_nodes"],configs["dropout_rate"], configs["l1_v"], configs["l2_v"])
        model.compile(optimizer=configs["optimizer"],
                      loss=configs["loss"],
                      metrics=configs["metrics"])


        model.fit(X_train, y_train, epochs=configs["epochs"], batch_size=configs["batch_size"], validation_freq=10, validation_split=0.2,callbacks=[wandb_callback])
        score = model.evaluate(X_test, y_test, verbose = 0,return_dict=True)
        for key,value in score.items():
            wandb.log({f"eval_{key}": value})
        wandb.finish()

        for count,(train,test) in enumerate(kfold.split(X,Y)):
            group_name = f"crossfold_random_state_{random_state}_{num_nodes}_dropout_{dropout_rate}_l1_{l1_v}"
            if epochs == 200:
                group_name = "200_epochs_" + group_name
            wandb.init(project=f"choose-best-config-central_{data_name}_gen_expr", config=configs,group=f"crossfold_random_state_{random_state}_{num_nodes}_dropout_{dropout_rate}_l1_{l1_v}",job_type='train',name=f"k_fold_{count}")
            # client_dataset = tf.data.Dataset.from_tensor_slices((X.iloc[train], Y[train]))
            # Define WandbCallback for experiment tracking
            wandb_callback = WandbCallback(monitor='val_loss',
                                           log_weights=True,
                                           log_evaluation=True,
                                           save_model=False,
                                           save_weights_only=True)
            # dataset_size = len(list(client_dataset))
            # train_ds_size = int(0.8 * dataset_size)
            # valid_ds_size = int(0.2 * dataset_size)
            #
            # train_ds = client_dataset.take(train_ds_size).shuffle(10000,reshuffle_each_iteration=True).batch(configs["batch_size"]).repeat(configs["epochs"])
            # valid_ds= client_dataset.skip(train_ds_size)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.iloc[train])
            X_test = scaler.transform(X.iloc[test])
            model = get_seq_nn_model(X.iloc[train].shape[1], num_nodes,dropout_rate ,l1_v, configs["l2_v"])
            model.compile(optimizer=configs["optimizer"],
                          loss=configs["loss"],
                          metrics=configs["metrics"])


            #model.fit(train_ds,validation_freq=10,validation_data=(valid_ds),callbacks=[wandb_callback])
            model.fit(X_train, Y[train], epochs=configs["epochs"],batch_size=configs["batch_size"],callbacks=[wandb_callback])

            #evaluate utils
            score = model.evaluate(X_test, Y[test], verbose = 0,return_dict=True)

            for key,value in score.items():
                wandb.log({f"eval_{key}": value})
            wandb.finish()
