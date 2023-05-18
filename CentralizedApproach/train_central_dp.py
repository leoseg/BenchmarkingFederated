from keras.losses import CategoricalCrossentropy
import tensorflow as tf
from utils.data_utils import load_data, create_X_y_from_gen_df, load_gen_data_as_train_test_split,preprocess_data
from utils.models import get_model
from sklearn.model_selection import StratifiedKFold, train_test_split
import wandb
from wandb.keras import WandbCallback
from utils.config import configs
import argparse
from sklearn.preprocessing import StandardScaler
import tensorflow_privacy as tfp

#Script that trains the model with given input configs, one time with a train, validation test split and one time with a stratified kfold
parser = argparse.ArgumentParser(
        prog="train_model_wandb_gen_expr.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

parser.add_argument(
    "--num_nodes", type=int, help="num of nodes for each layer",default=configs.get("num_nodes")
)

parser.add_argument(
    "--dropout_rate", type=float, help="dropout rate",default=configs.get("dropout_rate")
)
parser.add_argument(
    "--l1_v",type=float,help="l1 kernel regularizer",default=configs.get("l1_v")
)
parser.add_argument(
    "--data_path", type=str, help="path of data to load",default=configs.get("data_path")
)
parser.add_argument(
    "--l2_v",type=float,help="l1 kernel regularizer",default=configs.get("l2_v")
)
# print help if no argument is specified
args = parser.parse_args()

#create train test data


data_path = args.data_path
data_name = data_path.split("/")[2].split(".")[0]
modelname = data_path.split("/")[-1].split(".")[0]
# Loads data and creates X, Y arrays
df = load_data(data_path)
df  = preprocess_data(df)
X, Y = create_X_y_from_gen_df(df, False,configs.get("label"))
random_state = 69
num_nodes = args.num_nodes
dropout_rate = args.dropout_rate
l1_v = args.l1_v
project_name = "central_dp_metrics"
for noise in configs.get("noises"):
    group_name=f"usecase_{configs['usecase']}_noise_{noise}"

    # Trains the model with a train, validation, test split
    wandb.init(project=project_name, config=configs,group=group_name,job_type='train',name=f"no_crossfold")
    wandb_callback = WandbCallback(monitor='val_loss',
                                   log_weights=True,
                                   log_evaluation=True,
                                   save_model=False,
                                   save_weights_only=True)

    X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=0.2, random_state=69,stratify=Y)
    if configs["scale"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    optimizer = tfp.DPKerasAdamOptimizer(l2_norm_clip=1.0,noise_multiplier=noise)
    model = get_model(input_dim=X_train.shape[1], num_nodes=num_nodes,dropout_rate=dropout_rate, l1_v=l1_v, l2_v=configs.get("l2_v"))
    model.compile(optimizer=optimizer,
                  loss=configs.get("dp_loss"),
                  metrics=configs.get("metrics"))

    history = model.fit(X_train, y_train, epochs=300, batch_size=configs.get("batch_size"), validation_freq=configs["valid_freq"], validation_split=0.1,callbacks=[wandb_callback])
    score = model.evaluate(X_test, y_test, verbose = 0,return_dict=True)


    for key,value in score.items():
        wandb.log({f"eval_{key}": value})
    wandb.finish()

    # Creates and loops trough all kfolds
    kfold = StratifiedKFold(n_splits=configs.get("n_splits"), shuffle=True, random_state=random_state)
    for count,(train,test) in enumerate(kfold.split(X,Y)):
        wandb.init(project=project_name, config=configs,group=group_name,job_type='train',name=f"k_fold_{count}")
        if configs.get("usecase") != 2:
            validation_steps = int(configs.get("epochs")/10)
        else:
            validation_steps = 1
        wandb_callback = WandbCallback(monitor='val_loss',
                                       log_weights=True,
                                       log_evaluation=True,
                                       save_model=False,
                                       save_weights_only=True)
        X_train = X.iloc[train]
        X_test =X.iloc[test]
        if configs.get("scale"):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        optimizer = tfp.DPKerasAdamOptimizer(l2_norm_clip=1.0, noise_multiplier=noise,num_microbatches=1)
        model = get_model(input_dim=X_train.shape[1], num_nodes=num_nodes,dropout_rate=dropout_rate, l1_v=l1_v, l2_v=configs.get("l2_v"))
        model.compile(optimizer,
                      loss=configs.get("loss"),
                      metrics=configs.get("metrics"))
        model.fit(X_train, Y[train], epochs=100,batch_size=configs.get("batch_size"),callbacks=[wandb_callback])

        #evaluate
        score = model.evaluate(X_test, Y[test], verbose = 0,return_dict=True)
        for key,value in score.items():
            wandb.log({f"eval_{key}": value})
        wandb.finish()
