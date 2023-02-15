import os

from keras.metrics import AUC,Precision,Recall,BinaryAccuracy
from keras.optimizers import Adam,SGD
from keras.losses import BinaryCrossentropy
tff_time_logging_directory = "timelogs/tff_logs_time.txt"
flw_time_logging_directory = "timelogs/flw_logs_time.txt"
DATA_PATH = ""
if os.environ["USECASE"] == 2:
    configs = dict(
        usecase=1,
        batch_size=512,
        epochs=2,
        optimizer=SGD(),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy(), AUC(name="auc"), Precision(name="precision"), Recall(name="recall")],
        l2_v=0.001,
        n_splits=5,
        data_path="../DataGenExpression/Alldata.csv",
        shuffle=10000,
        label="Condition",
        scale=True
    )
else:
    configs = dict(
        usecase = 1,
        batch_size = 512,
        epochs = 100,
        optimizer = Adam(),
        loss = BinaryCrossentropy(),
        metrics = [BinaryAccuracy(),AUC(name="auc"),Precision(name="precision"),Recall(name="recall")],
        earlystopping_patience = 5,
        num_nodes = 1024,
        dropout_rate = 0.3,
        l1_v = 0.0,
        l2_v = 0.005,
        n_splits = 5,
        data_path="../DataGenExpression/Alldata.csv",
        shuffle=10000,
        label="Condition",
        scale=True
    )