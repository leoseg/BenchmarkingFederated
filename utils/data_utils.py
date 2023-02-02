import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from config import configs
import tensorflow as tf
kfold = StratifiedKFold(n_splits=configs["n_splits"],shuffle=True,random_state=0)


def clean_genexpr_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data and formats data
    :param df: dataframe to clean
    :return: cleaned pandas dataframe
    """
    df = df.drop(columns=['Dataset', 'GSE', 'Disease', 'Tissue', 'FAB', 'Filename', 'FAB_all'])
    df.Condition = df.Condition.map(dict(CASE=1, CONTROL=0))
    df = df.astype('int64')
    df = df.dropna()
    return df


def create_X_y(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Gets X and y from dataframe and scales X
    :param df: dataframe with data
    :return: X,y dataframes
    """
    X = df.drop(['Condition'], axis=1)
    y = df['Condition']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    return X, y


def load_gen_data_as_train_test_split(data_path: str,rows_to_keep=None):
    """
    Loads data from the given path and processes it to test,train format
    :param data_path: path where data is
    :return: train test arrays
    """
    df = load_gen_data(data_path,rows_to_keep)
    X, y = create_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    return X_train, X_test, y_train, y_test


def load_gen_data(data_path: str,rows_to_keep= None):
    """
    Load data from given path and preprocesses it
    :param data_path: path to data
    :return: datafrane
    """
    if rows_to_keep:
        df = pd.read_csv(data_path, skiprows=lambda x: x not in rows_to_keep)
    else:
        df = pd.read_csv(data_path)
    df = df.rename(columns={'Unnamed: 0': 'Sample'})
    df = df.set_index("Sample")
    df = clean_genexpr_data(df)
    return df

def load_gen_data_as_train_test_dataset_balanced(data_path:str, rows_to_keep=None, kfold_num:int=0):
    """
    Loads gen data from given path and splits is into train and test dataset
    :param data_path: path to gen data file
    :param rows_to_keep: if given only keep this rows of data file
    :param kfold_num: used too choose which fold to use for test and train
    :return:
    """
    df = load_gen_data(data_path,rows_to_keep)
    X, Y = create_X_y(df)
    for count, (train, test) in enumerate(kfold.split(X, Y)):
        if count == kfold_num:
            train_dataset = tf.data.Dataset.from_tensor_slices((X.iloc[train], Y[train]))
            test_dataset = tf.data.Dataset.from_tensor_slices(X.iloc[test],Y[test])
            return train_dataset,test_dataset

def preprocess(dataset : tf.data.Dataset,epochs :int = configs["epochs"]):
        return dataset.shuffle(configs["shuffle"], seed =1,reshuffle_each_iteration=True).batch(configs["batch_size"]).repeat(epochs)


def create_class_balanced_partitions(data_path:str, num_partitions:int):
    partitioner = StratifiedKFold(n_splits=num_partitions,shuffle=True,random_state=0)
    partition_rows = []
    df = load_gen_data(data_path)
    X,y = create_X_y(df)
    for _,rows in partitioner.split(X,y):
        partition_rows.append(rows)
    return partition_rows