import pandas as pd
import numpy as np



def read_file():
    train_file="train.csv"
    data=pd.read_csv(train_file)
    x_train=data.values[:,0:53]
    y_train=data.values[:,53:54]
    y_train=y_train.transpose()[0]

    y_train1=[]
    for i in y_train:

        if i==1:
            y_train1.append(1.0)
        else:
            y_train1.append(-1.0)

    return x_train,y_train1

#read_file()