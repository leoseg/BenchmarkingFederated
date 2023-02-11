from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from TensorflowFederated.data_loading import GenDataBackend
from data_utils import create_unbalanced_splits
import pandas as pd
data_path ="../DataGenExpression/Dataset1.csv"
# df = pd.read_csv("class_num.csv")
# fig = df.plot(kind="bar", stacked=True)
# plt.xlabel = "Clients"
# plt.ylabel = "Num examples"
# plt.title = "Examples per class and per client"
# plt.show()
# plt.savefig("test.png")
create_unbalanced_splits(data_path,"Condition",unweight_step=6)
# X,y = create_X_y(df)
# print(len(y))
# for i in range(3,11):
#     partitioner = StratifiedKFold(n_splits=i,shuffle=True)
#     partition_rows = []
#     for _,rows in partitioner.split(X,y):
#         rows = list(numpy.asarray(rows) + 1)
#         rows.append(0)
#         partition_rows.append(rows)
#
#     backend = GenDataBackend(partition_rows[0],kfold_num=1,data_path=data_path,local_epochs=100,random_state=1)
#     data = backend.materialize()
#     print(sum([len(list) for list in partition_rows]))

