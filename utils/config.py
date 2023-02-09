import os
from keras.metrics import AUC,Precision,Recall,BinaryAccuracy
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
jobid = os.environ["SLURM_ARRAY_TASK_ID"]
tff_time_logging_directory = f"timelogs/tff_logs_time_{jobid}.txt"
flw_time_logging_directory = f"timelogs/flw_logs_time_{jobid}.txt"
path_to_partitionlist = f"partitions_list_{jobid}"
DATA_PATH = ""
configs = dict(
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
    data_path="../DataGenExpression/Dataset1.csv",
    shuffle=10000
)