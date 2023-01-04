from model import model
import os
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
from preprocessing import create_X_y,clean_genexpr_data
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
df = pd.read_csv(data_path)
df = clean_genexpr_data(df)
X,y = create_X_y(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

#get model
model = model(num_nodes,dropout_rate,l1_v,l2_v)
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

# train model
os.makedirs("models",exist_ok=True)
callbacks = [EarlyStopping(monitor='loss', patience=25),
         ModelCheckpoint(filepath='/models/genexpr_model_fit.h5', monitor='loss', save_best_only=True)]
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

#evaluate model
score = model.evaluate(X_test, y_test, verbose = 0)

with open('readme.txt', 'w+') as f:
    f.writelines(f"Test loss {score[0]}")
    f.writelines(f"Text accuracy {score[1]}")

print('Test loss:', score[0])
print('Test accuracy:', score[1])