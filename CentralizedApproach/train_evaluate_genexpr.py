from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.data_utils import load_gen_data, create_X_y
from utils.models import get_seq_nn_model
from keras.metrics import Precision,Recall
from sklearn.model_selection import StratifiedKFold

#params
optimizer = "adam"
loss = "binary_crossentropy"
metrics =["accuracy","AUC",Precision(),Recall()]
num_nodes = 1024
dropout_rate = 0.3
l1_v = 0.0
l2_v = 0.005
epochs = 100
batch_size = 512


#create train test data
data_path ="../DataGenExpression/Alldata.csv"
modelname = data_path.split("/")[-1].split(".")[0]
df = load_gen_data(data_path)
X, Y= create_X_y(df)
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
#get utils
for train,test in kfold.split(X,Y):
    model = get_seq_nn_model(X[train].shape[1], num_nodes, dropout_rate, l1_v, l2_v)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    # train utils
    # os.makedirs("models",exist_ok=True)
    callbacks = [EarlyStopping(monitor='loss', patience=25),
             ModelCheckpoint(filepath=f'models/genexpr_model_{modelname}.h5', monitor='loss', save_best_only=True)]
    model.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    #evaluate utils
    score = model.evaluate(X[test], Y[test], verbose = 0,return_dict=True)

    # with open('readme.txt', 'a+') as f:
    #     f.writelines(f"Test loss {modelname} {score[0]}")
    #     f.writelines(f"Text accuracy {modelname} {score[1]}")
    for key,value in score.items():
        print(f"{key} : {value}")
