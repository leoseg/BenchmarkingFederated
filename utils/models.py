import keras
from keras.layers import Dense, Dropout,Input
from keras.models import Sequential
from keras.regularizers import l1_l2, l2
from config import configs
import visualkeras
from keras.utils import plot_model

def get_model(**kwargs)->keras.Model:
    """
    Uses model depending on the configs
    :param **kwargs for model
    :return: model
    """
    if configs.get("usecase") == 1:
        return get_seq_nn_model(input_dim=kwargs["input_dim"],num_nodes=kwargs["num_nodes"],dropout_rate=kwargs["dropout_rate"],l1_v=kwargs["l1_v"],l2_v=kwargs["l2_v"])
    elif configs.get("usecase") == 2:
        return get_log_reg_keras(l2_v=kwargs["l2_v"])
    elif configs.get("usecase") == 3:
        return get_log_reg_keras(l2_v=kwargs["l2_v"])






param_num_nodes = 1024
param_dropout_rate = 0.3
param_l1_v = 0.0
param_l2_v = 0.005
kernel_initialzier = 'glorot_uniform'
def get_seq_nn_model(input_dim:int,num_nodes: int = param_num_nodes, dropout_rate: float = param_dropout_rate, l1_v: float = param_l1_v,
                     l2_v: float = param_l2_v):
    """
    Creates model with given parameters
    :param input_dim: input dimension of data
    :param num_nodes: number of nodes in dense layers
    :param dropout_rate: dropout rate after denser layer
    :param l1_v: l1 kernel regularizer
    :param l2_v: l2 kernel regularizer
    :return: uncompiled utils
    """
    model = Sequential()
    # input layer
    model.add(Dense(256, activation='relu',kernel_initializer=kernel_initialzier , kernel_regularizer=l1_l2(l1=0.0, l2=0.0), input_dim=input_dim))
    model.add(Dropout(0.4))

    # first layer
    model.add(Dense(num_nodes, activation='relu', kernel_initializer=kernel_initialzier ,kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # second layer
    model.add(Dense(int(num_nodes / 2), activation='relu',kernel_initializer=kernel_initialzier , kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # third layer
    model.add(Dense(int(num_nodes / 2), activation='relu', kernel_initializer=kernel_initialzier ,kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # fourth layer
    model.add(Dense(int(num_nodes / 4), activation='relu',kernel_initializer=kernel_initialzier , kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # fifth layer
    model.add(Dense(int(num_nodes / 4), activation='relu',kernel_initializer=kernel_initialzier , kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # sixth layer
    model.add(Dense(int(num_nodes / 8), activation='relu', kernel_initializer=kernel_initialzier ,kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # seventh layer
    model.add(Dense(int(num_nodes / 8), activation='relu',kernel_initializer=kernel_initialzier , kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # eighth layer
    model.add(Dense(int(num_nodes / 16), activation='relu', kernel_initializer=kernel_initialzier ,kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(units=1, activation="tanh"))
    return model


def get_log_reg_keras(l2_v):
    """
    Returns log regression model implemented in keras
    :param l2_v: regularization in l2
    :return: model
    """
    model = Sequential()
    #model.add(Input(shape=(configs["batch_size"],input_dim)))
    model.add(Dense(input_dim=configs["input_dim"],units=configs["number_of_classes"], kernel_initializer='glorot_uniform', activation=configs["activation"], kernel_regularizer=l2(l2_v)))
    return model


