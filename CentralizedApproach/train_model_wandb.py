import datetime

from utils.data_utils import load_gen_data, create_X_y, load_gen_data_as_train_test_split
from utils.models import get_seq_nn_model
from keras.metrics import Precision,Recall,AUC
from sklearn.model_selection import StratifiedKFold
import wandb
import tensorflow as tf
from wandb.keras import WandbCallback


configs = dict(
    batch_size = 512,
    epochs = 100,
    optimizer = 'adam',
    loss = "binary_crossentropy",
    metrics = ["accuracy",AUC(curve="PR"),Precision(),Recall()],
    earlystopping_patience = 5,
    num_nodes = 1024,
    dropout_rate = 0.3,
    l1_v = 0.0,
    l2_v = 0.005,
    n_splits = 5

)
#create train test data
data_path ="../DataGenExpression/Dataset1.csv"
data_name = data_path.split("/")[2].split(".")[0]
modelname = data_path.split("/")[-1].split(".")[0]
df = load_gen_data(data_path)
X, Y= create_X_y(df)
kfold = StratifiedKFold(n_splits=configs["n_splits"],shuffle=True,random_state=0)


for count,(train,test) in enumerate(kfold.split(X,Y)):
    wandb.init(project=f"benchmark-central_{data_name}_with_tf_dataset", config=configs, job_type='train',name=f"k_fold_{count}")

    client_dataset = tf.data.Dataset.from_tensor_slices((X.iloc[train], Y[train])).shuffle(10000)
    # Define WandbCallback for experiment tracking
    wandb_callback = WandbCallback(monitor='val_loss',
                                   log_weights=True,
                                   log_evaluation=True,
                                   validation_steps=5,
                                   save_model=False,
                                   save_weights_only=True)
    dataset_size = len(list(client_dataset))
    train_ds_size = int(0.8 * dataset_size)
    valid_ds_size = int(0.2 * dataset_size)

    train_ds = client_dataset.take(train_ds_size).shuffle(10000,reshuffle_each_iteration=True).batch(configs["batch_size"]).repeat(configs["epochs"])
    valid_ds= client_dataset.skip(train_ds_size)
    model = get_seq_nn_model(X.iloc[train].shape[1], configs["num_nodes"],configs["dropout_rate"], configs["l1_v"], configs["l2_v"])
    model.compile(optimizer=configs["optimizer"],
                  loss=configs["loss"],
                  metrics=configs["metrics"])

    model.fit(train_ds,validation_freq=10,validation_data=(valid_ds),callbacks=[wandb_callback])
    #model.fit(X.iloc[train], Y[train], epochs=configs["epochs"], batch_size=configs["batch_size"], validation_freq = 10, validation_split=0.2,callbacks=[wandb_callback])

    #evaluate utils
    score = model.evaluate(X.iloc[test], Y[test], verbose = 0,return_dict=True)
    # with open('readme.txt', 'a+') as f:
    #     f.writelines(f"Test loss {modelname} {score[0]}")
    #     f.writelines(f"Text accuracy {modelname} {score[1]}")

    for key,value in score.items():
        wandb.log({f"eval_{key}": value})
    wandb.finish()
