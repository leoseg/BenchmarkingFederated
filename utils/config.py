import os
from metrics import AUC as SparseAUC
from keras.metrics import AUC,Precision,Recall,BinaryAccuracy,CategoricalCrossentropy,SparseCategoricalAccuracy
from keras.optimizers import Adam,SGD
from keras.losses import BinaryCrossentropy,SparseCategoricalCrossentropy
tff_time_logging_directory = "timelogs/tff_logs_time.txt"
flw_time_logging_directory = "timelogs/flw_logs_time.txt"
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
        metrics=[BinaryAccuracy(), AUC(name="auc"), Precision(name="precision"), Recall(name="recall")],
        earlystopping_patience=5,
        num_nodes=1024,
        dropout_rate=0.3,
        l1_v=0.0,
        l2_v=0.005,
        n_splits=5,
        data_path="../DataGenExpression/Dataset1.csv",
        shuffle=10000,
        label="Condition",
        scale=True,
        input_dim=12708,
        number_of_classes=1
    )
elif os.environ["USECASE"] == str(4):
    configs = dict(
        activation="softmax",
        random_state_partitions =69,
        valid_freq = 10,
        usecase = 4,
        batch_size = 512,
        epochs = 30,
        optimizer = Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy(), SparseAUC(name="auc"), SparseAUC(curve="PR", name="prauc")],
        earlystopping_patience = 5,
        num_nodes = 512,
        dropout_rate = 0.15,
        l1_v = 0.0,
        l2_v = 0.005,
        n_splits = 5,
        data_path="../Dataset2/Braindata_five_classes.csv",
        shuffle=10000,
        label="Classification",
        scale=False,
        categorical=True,
        input_dim=1426,
        number_of_classes=5,
    )
elif os.environ["USECASE"] == str(3):
    configs = dict(
        activation="softmax",
        valid_freq=2,
        usecase=3,
        batch_size=512,
        epochs=10,
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy(),SparseAUC(name="auc"),SparseAUC(curve="PR",name="prauc")],
        l2_v=1.0,
        n_splits=5,
        shuffle=10000,
        label="Classification",
        scale=False,
        number_of_classes=5,
        random_state_partitions=69,
        categorical=True,
        data_path="../Dataset2/Braindata_five_classes.csv",
        input_dim=1426,
    )
elif os.environ["USECASE"] == str(2):
    configs = dict(
        activation="sigmoid",
        valid_freq=2,
        usecase=2,
        batch_size=512,
        epochs=8,
        optimizer=SGD(),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy(), AUC(name="auc"), Precision(name="precision"), Recall(name="recall")],
        l2_v=0.001,
        n_splits=5,
        data_path="../DataGenExpression/Alldata.csv",
        shuffle=10000,
        label="Condition",
        scale=True,
        number_of_classes=1,
        random_state_partitions=69,
        input_dim=12708,
    )
else:
    configs = dict(
        activation = "sigmoid",
        random_state_partitions =69,
        valid_freq = 10,
        usecase = 1,
        batch_size = 512,
        epochs = 70,
        optimizer = Adam(),
        loss = BinaryCrossentropy(),
        metrics = [BinaryAccuracy(),AUC(name="auc"),Precision(name="precision"),Recall(name="recall")],
        earlystopping_patience = 5,
        num_nodes = 512,
        dropout_rate = 0.15,
        l1_v = 0.0,
        l2_v = 0.005,
        n_splits = 5,
        data_path="../DataGenExpression/Alldata.csv",
        shuffle=10000,
        label="Condition",
        scale=True,
        input_dim=12708,
        number_of_classes =1
    )