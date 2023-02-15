import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from dtale import show
df = pd.read_csv("Maternal Health Risk Data Set.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
show(df)
train, test = train_test_split(df, test_size=0.2)
X = df.iloc[1:, 0:-1]
y = df.iloc[1:, -1]
X_train, X_test,y_train,y_test= train_test_split(X,y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
for c in [0.001,0.01,0.1,1,10,100]:
    lr = SGDClassifier(loss="log_loss",max_iter=100000,penalty="l2",class_weight="balanced")
    lr.fit(X_train,y_train)
    print(lr.score(X_test,y_test))
