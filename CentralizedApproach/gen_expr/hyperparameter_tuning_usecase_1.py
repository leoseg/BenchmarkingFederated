from utils.data_utils import load_gen_data_as_train_test_split
from utils.models import get_seq_nn_model
from utils.config import configs
import wandb
from wandb.keras import WandbCallback

# This script was used for tuning the hyper parameter for the first usecase

sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "sum_accuracy"},
    "parameters": {
        "num_nodes": {"values": [256, 512, 1024]},
        "dropout_rate": {"values": [0.15, 0.3, 0.5]},
        "l1_v": {"values": [0.005, 0.0]},
        "l2_v": {"values": [0.0, 0.005]},
    },
}
data_path = "../../DataGenExpression/Alldata.csv"
data_name = data_path.split("/")[2].split(".")[0]
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project=f"sweep_usecase_{configs['usecase']}"
)
# create train test data

modelname = data_path.split("/")[-1].split(".")[0]
X_train, X_test, y_train, y_test = load_gen_data_as_train_test_split(data_path)


# get utils
def train():
    """
    Train function for wandb sweep
    """
    wandb.init(
        project=f"sweep_usecase_{configs['usecase']}", config=configs, job_type="train"
    )
    ACCUMULATED_METRICS = {}
    # Define WandbCallback for experiment tracking
    wandb_callback = WandbCallback(
        monitor="val_loss",
        log_weights=True,
        log_evaluation=True,
        validation_steps=5,
        save_model=False,
        save_weights_only=True,
    )
    config = wandb.config
    model = get_seq_nn_model(
        X_train.shape[1],
        config.get("num_nodes"),
        config.get("dropout_rate"),
        config.get("l1_v"),
        config.get("l2_v"),
    )
    model.compile(
        optimizer=configs.get("optimizer"),
        loss=configs.get("loss"),
        metrics=configs.get("metrics"),
    )

    model.fit(
        X_train,
        y_train,
        epochs=configs.get("epochs"),
        batch_size=configs.get("batch_size"),
        validation_freq=10,
        validation_split=0.2,
        callbacks=[wandb_callback],
    )

    # evaluate utils
    score = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

    for key, value in score.items():
        wandb.log({f"eval_{key}": value})
    wandb.finish()


wandb.agent(sweep_id, function=train, count=10)
