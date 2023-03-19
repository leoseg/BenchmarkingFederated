from utils.data_utils import load_data, create_X_y_from_gen_df, preprocess_data
from utils.models import get_seq_nn_model
from sklearn.model_selection import StratifiedKFold
import wandb
from wandb.keras import WandbCallback
from utils.config import configs
import argparse
import pandas
# Script to see what happens if while training in each kfold there is data from all three datasets of usecase 1 which form the entire dataset
parser = argparse.ArgumentParser(
        prog="train_model_wandb_gen_expr_take_data_from_all.py",
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
# print help if no argument is specified
args = parser.parse_args()

#create train test data
data_pathes = ["../DataGenExpression/Dataset1.csv","../DataGenExpression/Dataset2.csv","../DataGenExpression/Dataset3.csv"]
Xs= []
Ys=[]
for data_path in data_pathes:
    df = load_data(data_path)
    df = preprocess_data(df)
    X, Y = create_X_y_from_gen_df(df,label=configs.get("label"))
    Xs.append(X)
    Ys.append(Y)
random_state = 69
kfold = StratifiedKFold(n_splits=configs.get("n_splits"), shuffle=True, random_state=random_state)

num_nodes = args.num_nodes
dropout_rate = args.dropout_rate
l1_v = args.l1_v


for count,((train,test),(train2,test2),(train3,test3)) in enumerate(zip(kfold.split(Xs[0],Ys[0]),kfold.split(Xs[1],Ys[1]),kfold.split(Xs[2],Ys[2]))):
    X_train = pandas.concat([Xs[0].iloc[train],Xs[1].iloc[train2],Xs[2].iloc[train3]],ignore_index=True)
    Y_train = pandas.concat([Ys[0][train],Ys[1][train2],Ys[2][train3]],ignore_index=True)
    X_test= pandas.concat([Xs[0].iloc[test], Xs[1].iloc[test2], Xs[2].iloc[test3]],ignore_index=True)
    Y_test = pandas.concat([Ys[0][test], Ys[1][test2], Ys[2][test3]],ignore_index=True)

    wandb.init(project=f"choose-best-config-central_Alldata_gen_expr",config=configs,group=f"even_split_crossfold_random_state_{random_state}_{num_nodes}_dropout_{dropout_rate}_l1_{l1_v}",job_type='train',name=f"k_fold_{count}")

    wandb_callback = WandbCallback(monitor='val_loss',
                                   log_weights=True,
                                   log_evaluation=True,
                                   save_model=False,
                                   save_weights_only=True)
    model = get_seq_nn_model(X_train.iloc[train].shape[1], num_nodes,dropout_rate ,l1_v, configs.get("l2_v"))
    model.compile(optimizer=configs.get("optimizer"),
                  loss=configs.get("loss"),
                  metrics=configs.get("metrics"))
    model.fit(X_train, Y_train, epochs=configs.get("epochs"),batch_size=configs.get("batch_size"),callbacks=[wandb_callback])

    #evaluate utils
    score = model.evaluate(X_test,Y_test, verbose = 0,return_dict=True)

    for key,value in score.items():
        wandb.log({f"eval_{key}": value})
    wandb.finish()