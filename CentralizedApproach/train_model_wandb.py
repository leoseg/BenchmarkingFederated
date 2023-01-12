from utils.data_utils import load_gen_data, create_X_y
from utils.models import get_seq_nn_model
from keras.metrics import Precision,Recall
from sklearn.model_selection import StratifiedKFold
import wandb
from tensorflow_addons.metrics import F1Score
from wandb.keras import WandbCallback


configs = dict(
    batch_size = 512,
    epochs = 100,
    optimizer = 'adam',
    loss = "binary_crossentropy",
    metrics = ["accuracy","AUC",Precision(),Recall()],
    earlystopping_patience = 5,
    num_nodes = 1024,
    dropout_rate = 0.3,
    l1_v = 0.0,
    l2_v = 0.005,
    n_splits =2

)
#create train test data
data_path ="../DataGenExpression/Dataset1.csv"
data_name = data_path.split("/")[2].split(".")[0]
modelname = data_path.split("/")[-1].split(".")[0]
df = load_gen_data(data_path)
X, Y= create_X_y(df)
kfold = StratifiedKFold(n_splits=configs["n_splits"],shuffle=True,random_state=0)

#get utils

ACCUMULATED_METRICS = {}
for count,(train,test) in enumerate(kfold.split(X,Y)):
    wandb.init(project=f"benchmark-central_{data_name}", config=configs, job_type='train',name=f"k_fold_{count}")


    # Define WandbCallback for experiment tracking
    wandb_callback = WandbCallback(monitor='val_loss',
                                   log_weights=True,
                                   log_evaluation=True,
                                   validation_steps=5)
    model = get_seq_nn_model(X.iloc[train].shape[1], configs["num_nodes"],configs["dropout_rate"], configs["l1_v"], configs["l2_v"])
    model.compile(optimizer=configs["optimizer"],
                  loss=configs["loss"],
                  metrics=configs["metrics"])


    model.fit(X.iloc[train], Y[train], epochs=configs["epochs"], batch_size=configs["batch_size"], validation_freq = 10, validation_split=0.2,callbacks=[wandb_callback])

    #evaluate utils
    score = model.evaluate(X.iloc[test], Y[test], verbose = 0,return_dict=True)
    # with open('readme.txt', 'a+') as f:
    #     f.writelines(f"Test loss {modelname} {score[0]}")
    #     f.writelines(f"Text accuracy {modelname} {score[1]}")

    for key,value in score.items():
        wandb.log({f"eval_{key}": value})
        if ACCUMULATED_METRICS.get(key):
            ACCUMULATED_METRICS[key] = ACCUMULATED_METRICS[key] + value
        else:
            ACCUMULATED_METRICS[key] = value
    wandb.finish()
ACCUMULATED_METRICS = { key : value/configs["n_splits"] for (key, value) in ACCUMULATED_METRICS}
wandb.init(project ="benchmark-central", config= configs, job_type="train", name="summary")
for key,value in ACCUMULATED_METRICS:
    wandb.log(f"sum_{key} : {value}")
wandb.finish()