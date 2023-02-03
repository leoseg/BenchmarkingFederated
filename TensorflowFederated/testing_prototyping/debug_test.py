from data_loading import GenDataBackend
import pickle
from data_utils import create_class_balanced_partitions
lists = create_class_balanced_partitions(data_path="../../DataGenExpression/Dataset1.csv", num_partitions=2)
backend  = GenDataBackend(data_path="../../DataGenExpression/Dataset1.csv", rows_to_keep=lists[1], kfold_num=0, random_state=0, local_epochs=100)
