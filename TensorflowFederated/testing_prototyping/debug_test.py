from sklearn.model_selection import StratifiedKFold
from TensorflowFederated.data_loading import GenDataBackend
from data_utils import load_gen_data,create_X_y
import numpy
data_path ="../DataGenExpression/Alldata.csv"

df = load_gen_data(data_path)
X,y = create_X_y(df)
print(len(y))
for i in range(3,11):
    partitioner = StratifiedKFold(n_splits=i,shuffle=True)
    partition_rows = []
    for _,rows in partitioner.split(X,y):
        rows = list(numpy.asarray(rows) + 1)
        rows.append(0)
        partition_rows.append(rows)

    backend = GenDataBackend(partition_rows[0],kfold_num=1,data_path=data_path,local_epochs=100,random_state=1)
    data = backend.materialize()
    print(sum([len(list) for list in partition_rows]))

