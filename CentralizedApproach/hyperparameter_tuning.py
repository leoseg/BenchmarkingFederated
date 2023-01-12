from utils.data_utils import load_gen_data_as_train_test_split
from utils.models import get_seq_nn_model
from keras.metrics import Precision,Recall, AUC
from tensorflow_addons.metrics import F1Score
import wandb
from wandb.keras import WandbCallback


configs = dict(
    batch_size = 512,
    epochs = 100,
    optimizer = 'adam',
    loss = "binary_crossentropy",
    metrics = ["accuracy",AUC(),Precision(),Recall()],
    earlystopping_patience = 5,
    num_nodes = 1024,
    dropout_rate = 0.3,
    l1_v = 0.0,
    l2_v = 0.005,
    n_splits =5

)
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'sum_accuracy'},
    'parameters':
    {
        "num_nodes" : {"values": [256,512,1024]},
        "dropout_rate" : {"values" : [0.15,0.3,0.5]},
        "l1_v" : {"values" : [0.005,0.0]},
        "l2_v" : {"values" : [0.0,0.005]}
     }
}
data_path ="../DataGenExpression/Dataset1.csv"
data_name = data_path.split("/")[2].split(".")[0]
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'benchmark-central_sweep_{data_name}')
#create train test data

modelname = data_path.split("/")[-1].split(".")[0]
X_train, X_test, y_train, y_test = load_gen_data_as_train_test_split(data_path)


#get utils
def train():
    wandb.init(project=f"benchmark-central_sweep_{data_name}", config=configs, job_type='train')
    ACCUMULATED_METRICS = {}
    # Define WandbCallback for experiment tracking
    wandb_callback = WandbCallback(monitor='val_loss',
                                   log_weights=True,
                                   log_evaluation=True,
                                   validation_steps=5)

    model = get_seq_nn_model(X_train.shape[1], configs["num_nodes"],configs["dropout_rate"], configs["l1_v"], configs["l2_v"])
    model.compile(optimizer=configs["optimizer"],
                  loss=configs["loss"],
                  metrics=configs["metrics"])


    model.fit(X_train, y_train, epochs=configs["epochs"], batch_size=configs["batch_size"], validation_freq=10, validation_split=0.2,callbacks=[wandb_callback])

    #evaluate utils
    score = model.evaluate(X_test, y_test, verbose = 0,return_dict=True)

    for key,value in score.items():
        wandb.log({f"eval_{key}": value})
    wandb.finish()

wandb.agent(sweep_id, function=train, count=2)