import os
import tensorflow as tf
from metrics import AUC as SparseAUC
from keras.metrics import (
    AUC,
    Precision,
    Recall,
    BinaryAccuracy,
    CategoricalCrossentropy,
    SparseCategoricalAccuracy,
)
from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy

tff_time_logging_directory = "timelogs/tff_logs_time.txt"
flw_time_logging_directory = "timelogs/flw_logs_time.txt"
SEED = 42
version = "dp_noises"
dp_groups = [
    "flwr_5_0.1",
    "flwr_5_0.085",
    "flwr_5_0.07",
    "flwr_5_0.05",
    "flwr_5_0.03",
    "flwr_5_0.01",
    "tff_5_0.1",
    "tff_5_0.085",
    "tff_5_0.07",
    "tff_5_0.05",
    "tff_5_0.03",
    "tff_5_0.01",
]
n_splits = 5
noises = [0.01, 0.03, 0.05, 0.07, 0.085, 0.1]
DATA_PATH = ""
if os.environ["USECASE"] == "test":
    configs = dict(
        activation="sigmoid",
        random_state_partitions=69,
        valid_freq=10,
        usecase=1,
        batch_size=512,
        epochs=100,
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        dp_loss=BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
        metrics=[
            BinaryAccuracy(),
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
        earlystopping_patience=5,
        num_nodes=1024,
        dropout_rate=0.3,
        l1_v=0.0,
        l2_v=0.005,
        n_splits=n_splits,
        data_path="../DataGenExpression/Dataset1.csv",
        shuffle=10000,
        label="Condition",
        scale=True,
        input_dim=12708,
        number_of_classes=1,
        random_seed_set=True,
        version=version,
    )
elif os.environ["USECASE"] == str(4):
    configs = dict(
        plot_path="../../MA/pictures/scenario4/",
        activation="softmax",
        random_state_partitions=69,
        valid_freq=10,
        usecase=4,
        dp_loss=SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE, from_logits=True
        ),
        batch_size=512,
        epochs=30,
        global_norm=48.7,
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=[
            SparseCategoricalAccuracy(),
            SparseAUC(name="auc"),
            SparseAUC(curve="PR", name="prauc"),
        ],
        earlystopping_patience=5,
        groups=[
            "tff_3",
            "tff_5",
            "tff_10",
            "tff_50",
            "flwr_3",
            "flwr_5",
            "flwr_10",
            "flwr_50",
        ],
        unweighted_groups=[
            "tff_0.0",
            "tff_4.0",
            "tff_8.0",
            "tff_10.0",
            "tff_12.0",
            "tff_14.0",
            "tff_16.0",
            "flwr_0.0",
            "flwr_4.0",
            "flwr_8.0",
            "flwr_10.0",
            "flwr_12.0",
            "flwr_14.0",
            "flwr_16.0",
        ],
        dp_groups=dp_groups,
        num_nodes=512,
        dropout_rate=0.15,
        l1_v=0.0,
        l2_v=0.005,
        n_splits=n_splits,
        data_path="../Dataset2/Braindata_five_classes.csv",
        shuffle=10000,
        label="Classification",
        scale=False,
        categorical=True,
        input_dim=1426,
        number_of_classes=5,
        random_seed_set=True,
        version=version,
        data_directory="../Dataset2/",
        num_examples_10=554,
        delta=0.0005,
        dp_epochs=100,
        noises=noises,
    )
elif os.environ["USECASE"] == str(3):
    configs = dict(
        plot_path="../../MA/pictures/scenario3/",
        activation="softmax",
        valid_freq=2,
        usecase=3,
        batch_size=512,
        epochs=10,
        global_norm=3.35,
        optimizer=Adam(),
        groups=[
            "tff_3",
            "tff_5",
            "tff_10",
            "tff_50",
            "flwr_3",
            "flwr_5",
            "flwr_10",
            "flwr_50",
        ],
        unweighted_groups=[
            "tff_0.0",
            "tff_4.0",
            "tff_8.0",
            "tff_10.0",
            "tff_12.0",
            "tff_14.0",
            "tff_16.0",
            "flwr_0.0",
            "flwr_4.0",
            "flwr_8.0",
            "flwr_10.0",
            "flwr_12.0",
            "flwr_14.0",
            "flwr_16.0",
        ],
        dp_groups=dp_groups,
        loss=SparseCategoricalCrossentropy(),
        dp_loss=SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
        metrics=[
            SparseCategoricalAccuracy(),
            SparseAUC(name="auc"),
            SparseAUC(curve="PR", name="prauc"),
        ],
        l2_v=1.0,
        n_splits=n_splits,
        shuffle=10000,
        label="Classification",
        scale=False,
        number_of_classes=5,
        random_state_partitions=69,
        categorical=True,
        data_path="../Dataset2/Braindata_five_classes.csv",
        input_dim=1426,
        random_seed_set=True,
        version=version,
        data_directory="../Dataset2/",
        num_examples_10=554,
        delta=0.0005,
        noises=noises,
    )
elif os.environ["USECASE"] == str(2):
    configs = dict(
        plot_path="../../MA/pictures/scenario2/",
        activation="sigmoid",
        valid_freq=2,
        usecase=2,
        batch_size=512,
        epochs=8,
        groups=[
            "tff_3",
            "tff_5",
            "tff_10",
            "tff_50",
            "flwr_3",
            "flwr_5",
            "flwr_10",
            "flwr_50",
        ],
        unweighted_groups=[
            "tff_0.0",
            "tff_2.0",
            "tff_4.0",
            "tff_6.0",
            "tff_8.0",
            "tff_9.0",
            "tff_10.0",
            "flwr_0.0",
            "flwr_2.0",
            "flwr_4.0",
            "flwr_6.0",
            "flwr_8.0",
            "flwr_9.0",
            "flwr_10.0",
        ],
        dp_groups=dp_groups,
        optimizer=SGD(),
        global_norm=298.85,
        loss=BinaryCrossentropy(),
        dp_loss=BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
        metrics=[
            BinaryAccuracy(),
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(curve="PR", name="prauc"),
            AUC(multi_label=True, name="auc_macro"),
        ],
        l2_v=0.001,
        n_splits=n_splits,
        data_path="../DataGenExpression/Alldata.csv",
        shuffle=10000,
        label="Condition",
        scale=True,
        number_of_classes=1,
        random_state_partitions=69,
        input_dim=12708,
        random_seed_set=True,
        version=version,
        data_directory="../DataGenExpression/",
        num_examples_10=961,
        delta=0.0005,
        noises=noises,
    )
else:
    configs = dict(
        plot_path="../../MA/pictures/scenario1/",
        activation="sigmoid",
        random_state_partitions=69,
        valid_freq=10,
        usecase=1,
        batch_size=512,
        epochs=70,
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(),
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(curve="PR", name="prauc"),
            AUC(multi_label=True, name="auc_macro"),
        ],
        earlystopping_patience=5,
        global_norm=76.95,
        groups=[
            "tff_3",
            "tff_5",
            "tff_10",
            "tff_50",
            "flwr_3",
            "flwr_5",
            "flwr_10",
            "flwr_50",
        ],
        unweighted_groups=[
            "tff_0.0",
            "tff_2.0",
            "tff_4.0",
            "tff_6.0",
            "tff_8.0",
            "tff_9.0",
            "tff_10.0",
            "flwr_0.0",
            "flwr_2.0",
            "flwr_4.0",
            "flwr_6.0",
            "flwr_8.0",
            "flwr_9.0",
            "flwr_10.0",
        ],
        dp_groups=dp_groups,
        num_nodes=512,
        dropout_rate=0.15,
        l1_v=0.0,
        l2_v=0.005,
        n_splits=n_splits,
        data_path="../DataGenExpression/Alldata.csv",
        shuffle=10000,
        label="Condition",
        scale=True,
        input_dim=12708,
        number_of_classes=1,
        random_seed_set=True,
        version=version,
        dp_loss=BinaryCrossentropy(reduction="none"),
        data_directory="../DataGenExpression/",
        num_examples_10=961,
        delta=0.0005,
        noises=noises,
    )
