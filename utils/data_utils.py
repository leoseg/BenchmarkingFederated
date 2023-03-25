from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
from config import configs
import tensorflow as tf
import numpy
from math import floor
import numpy as np
from keras.utils import to_categorical


def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """
    Uses preprocess function depending on the configs
    :param df: df to preprocess
    :return: preprocessed df
    """
    if configs.get("usecase") == 1 or configs.get("usecase") == 2:
        return preprocess_genexpr_data(df)
    elif configs.get("usecase") == 3 or configs.get("usecase") == 4:
        if ("Unnamed: 0" or "Unnamed: 0.1") in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'Sample'})
        df = df.set_index("Sample")
        le = LabelEncoder()
        df[configs["label"]] = le.fit_transform(df[configs["label"]])
        return df


def select_feature_genes(df:pd.DataFrame, cov_threshold=3.5):
    """
    Preprocesses brain cell data by choosing most relevant genes by coefficient of variation filtering
    :param df: dataframe to filter
    :param cov_threshold: threshold for filtering genes
    :return: filtered df
    """
    df = df[df["Classification"] !="no class"]
    cell_types = sorted(set(list(df["Classification"])))
    median_df = pd.DataFrame(np.zeros((len(cell_types), df.shape[1])))  # 75 rows × 13945 columns
    median_df.columns = df.columns
    median_df.index = cell_types
    for classification in cell_types:
        subset = df.loc[df["Classification"] == classification]
        subset = subset.iloc[:, 0:subset.shape[1] - 1]
        median_df.loc[classification, :] = subset.median(axis=0)
    median_df = median_df.drop(columns=["Classification"], axis=1)
    median_df_mean = median_df.mean(axis=0)
    median_df_var = median_df.var(axis=0)
    median_df_cov = median_df_var / median_df_mean
    cov_threshold =cov_threshold
    cov_filtered = df[median_df_cov[median_df_cov > cov_threshold].index]
    cov_filtered["Classification"] = df["Classification"]
    cov_filtered.set_index(df["Unnamed: 0"],inplace=True)
    cov_filtered.to_csv("../Dataset2/Braindata.csv")
    return cov_filtered


def relabel_brain_cell_data(data_path:str, label_data_path, label:str=None):
    """
    Assigns clustered label to the brain cell dataframe
    :param data_path dataframe to relabel
    :param label_data_path: dataframe with index sample name and one column with new label
    :param label: label to use "brain_subregion", "class", "cluster"
    :return: new dataframe with label column classification
    """
    df = pd.read_csv(data_path)
    lable_columns = pd.read_csv(label_data_path)
    lable_columns.set_index(lable_columns['Sample'])
    df =lable_columns.join(df)
    df = df.set_index(df["Sample"])
    df = df.drop(columns=["Unnamed: 0", "Classification","Sample"])
    return df

def create_brain_cell_patient_partitions():
    """
    Splits brain data in indices for three partitions
    :return: list of partitions indices
    """
    sample_columns = pd.read_csv("../Dataset2/human_MTG_gene_expression_matrices_2018-06-14/human_MTG_2018-06-14_samples-columns.csv")
    sample_columns = sample_columns[sample_columns["label"] != "no class"]
    client_one_row_numbs = list(sample_columns[sample_columns["donor"]=="H200.1030"].index)
    client_one_row_numbs.append(0)
    client_two_row_numbs = list(sample_columns[sample_columns["donor"] == "HH200.1023"].index)
    client_two_row_numbs.append(0)
    client_three_row_numbs = list(sample_columns[sample_columns["donor"] != "HH200.1023" and sample_columns["donor"]!="H200.1030"].index)
    client_three_row_numbs.append(0)
    return [client_one_row_numbs,client_two_row_numbs,client_three_row_numbs]


def preprocess_genexpr_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data and formats data
    :param df: dataframe to clean
    :return: cleaned pandas dataframe
    """
    df = df.rename(columns={'Unnamed: 0': 'Sample'})
    df = df.set_index("Sample")
    columns_to_drop = ['Dataset', 'GSE', 'Disease', 'Tissue', 'FAB', 'Filename']
    if "FAB_all" in df.columns:
        columns_to_drop.append("FAB_all")
    df = df.drop(columns=columns_to_drop)
    df.Condition = df.Condition.map(dict(CASE=1, CONTROL=0))
    df = df.astype('int64')
    df = df.dropna()
    return df


def create_X_y_from_gen_df(df: pd.DataFrame, scaling=True,label=configs["label"]) -> (pd.DataFrame, pd.DataFrame):
    """
    Gets X and y from dataframe and scales X
    :param df: dataframe with data
    :return: X,y dataframes
    """
    #df = clean_genexpr_data(df)
    X = df.drop([label], axis=1)
    y = df[label]
    if scaling:
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
    df = load_data(data_path, rows_to_keep)
    df = preprocess_genexpr_data(df)
    X, y = create_X_y_from_gen_df(df, False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def load_data(data_path: str, rows_to_keep= None):
    """
    Load data from given path and preprocesses it
    :param data_path: path to data
    :return: datafrane
    """
    if rows_to_keep is not None:
        df = pd.read_csv(data_path, skiprows=lambda x: x not in rows_to_keep)
    else:
        df = pd.read_csv(data_path)
    return  df


def df_train_test_dataset(df: pd.DataFrame, kfold_num:int=0, random_state=0, label=configs["label"], scale=True):
    """
    Loads gen data from given path and splits is into train and test dataset
    :param data_path: path to gen data file
    :param rows_to_keep: if given only keep this rows of data file
    :param kfold_num: used to choose which fold to use for test and train
    :return:
    """
    kfold = StratifiedKFold(n_splits=configs.get("n_splits"), shuffle=True, random_state=random_state)
    #df = load_data(data_path, rows_to_keep)
    X, Y = create_X_y_from_gen_df(df, False,label)
    for count, (train, test) in enumerate(kfold.split(X, Y)):
        if count == kfold_num:
            X_train = X.iloc[train]
            X_test = X.iloc[test]
            if scale:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y[train]))
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y[test]))
            return train_dataset,test_dataset

def preprocess(dataset : tf.data.Dataset,epochs :int = configs.get("epochs")):
        return dataset.shuffle(configs.get("shuffle"), seed =1,reshuffle_each_iteration=True).batch(configs.get("batch_size")).repeat(epochs)


def create_class_balanced_partitions(df: pd.DataFrame, num_partitions:int,label=configs["label"]):
    """
    Created same sized partitions which all have the class balance from the orignal dataset
    :param df: Dataframe with whole dataset
    :param num_partitions: number of partitions to create
    :param label: label for classification
    :return: list of lists of indices which indicates which rows belonging to the partition
    """
    partitioner = StratifiedKFold(n_splits=num_partitions,shuffle=True,random_state=configs["random_state_partitions"])
    partition_rows = []
    #df = load_data(data_path)
    X, y = create_X_y_from_gen_df(df, False,label)
    for _,rows in partitioner.split(X,y):
        rows = list(numpy.asarray(rows) + 1)
        rows.append(0)
        partition_rows.append(rows)
    return partition_rows


def create_unbalanced_splits(df:pd.DataFrame,label_name:str,unweight_step:int):
    """
    Splits dataframe loaded with datapath into number of dataframes equal to number of partitions
    where each dataframe has one class weighted more than the others
    :param data_path: data path to original dataframe
    :param label_name: label of column to split dataframe by
    :param unweight_step: for each unweight step the choosen class has 5 percent more and the rest 5 percent less samples
    :return: dataframe with class number of examples per client
    """
    df = df.reset_index()
    class_percentages = df[label_name].value_counts(normalize=True)
    num_classes = len(class_percentages)
    partition_size = floor(min([ class_size * len(df) for class_size in class_percentages]))
    partitions_dict = defaultdict(list)
    clients = []
    #partitions_dfs = []
    start_percentage = 1.0/ num_classes
    partitions_list = []
    for partition_split in range(num_classes):
        #dfs = []
        clients.append(partition_split)
        indexes = []
        for count,(class_label,class_value) in enumerate(zip(class_percentages.index,class_percentages)):

            partition_value = 0.0
            if count == partition_split:
                partition_value  = start_percentage +0.05 * unweight_step if (start_percentage +0.05 * unweight_step)< 1.0 else 1.0
                sampled_index = df[df[label_name] == class_label].sample(floor(partition_value*partition_size),random_state=configs["random_state_partitions"]).index
                indexes.extend(list(sampled_index))
                df = df.drop(index=sampled_index)
                # dfs.append(sampled_df)
            else:
                partition_value  = start_percentage- 0.05/(num_classes-1) * unweight_step if  (start_percentage - 0.05/(num_classes-1) * unweight_step )> 0.0 else 0.0
                sampled_index = df[df[label_name] == class_label].sample(floor(partition_value * partition_size),random_state=configs["random_state_partitions"]).index
                indexes.extend(list(sampled_index))
                df = df.drop(index=sampled_index)
                # dfs.append(sampled_df)
            partitions_dict["class "+str(class_label)].append(floor(partition_value*partition_size))
        indexes.append(0)
        partitions_list.append(indexes)
        # partition_dataframe = pd.concat(dfs,ignore_index=True)
        # partitions_dfs.append(partition_dataframe)
        #partition_dataframe.to_csv(f"partition_{partition_split}.csv")

    return pd.DataFrame(partitions_dict,index=clients),partitions_list






