from utils.models import get_seq_nn_model
import os
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
from utils.data_utils import create_X_y,clean_genexpr_data,load_data
from sklearn.model_selection import train_test_split


#params
optimizer = "adam"
loss = "binary_crossentropy"
metrics =["accuracy"]
num_nodes = 1024
dropout_rate = 0.3
l1_v = 0.0
l2_v = 0.005
epochs = 100
batch_size = 512


#create train test data
data_path ="../DataGenExpression/Alldata.csv"
modelname = data_path.split("/")[-1].split(".")[0]
X_train, X_test, y_train, y_test = load_data(data_path)

#get utils
model = get_seq_nn_model(X_train.shape[1], num_nodes, dropout_rate, l1_v, l2_v)
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

# train utils
# os.makedirs("models",exist_ok=True)
callbacks = [EarlyStopping(monitor='loss', patience=25),
         ModelCheckpoint(filepath=f'models/genexpr_model_{modelname}.h5', monitor='loss', save_best_only=True)]
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

#evaluate utils
score = model.evaluate(X_test, y_test, verbose = 0)

with open('readme.txt', 'a+') as f:
    f.writelines(f"Test loss {modelname} {score[0]}")
    f.writelines(f"Text accuracy {modelname} {score[1]}")

print('Test loss:', score[0])
print('Test accuracy:', score[1])