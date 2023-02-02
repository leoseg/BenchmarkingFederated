from utils.data_utils import load_gen_data, create_X_y, load_gen_data_as_train_test_split
from utils.models import get_seq_nn_model
from keras.metrics import Precision,Recall,AUC
from sklearn.model_selection import StratifiedKFold
import wandb
import tensorflow as tf
from utils.data_utils import preprocess
from utils.config import configs


#create train test data
data_path =configs["data_path"]
data_name = data_path.split("/")[2].split(".")[0]
modelname = data_path.split("/")[-1].split(".")[0]
df = load_gen_data(configs["data_path"])
X, Y= create_X_y(df)
kfold = StratifiedKFold(n_splits=configs["n_splits"],shuffle=True,random_state=0)


for count,(train,test) in enumerate(kfold.split(X,Y)):
    wandb.init(project=f"benchmark-central_{data_name}_with_tf_dataset_and_time", config=configs, job_type='train',name=f"k_fold_{count}")

    client_dataset = preprocess(tf.data.Dataset.from_tensor_slices((X.iloc[train], Y[train])))



    model = get_seq_nn_model(X.iloc[train].shape[1], configs["num_nodes"],configs["dropout_rate"], configs["l1_v"], configs["l2_v"])
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
