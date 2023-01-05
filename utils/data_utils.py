import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_genexpr_data(df:pd.DataFrame)->pd.DataFrame:
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

def create_X_y(df:pd.DataFrame)->(pd.DataFrame,pd.DataFrame):
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
    return X,y


def load_data(data_path:str):
    """
    Loads data from the given path and processes it to test,train format
    :param data_path: path where data is
    :return: train test arrays
    """
    df = pd.read_csv(data_path)
    df = df.rename(columns={'Unnamed: 0': 'Sample'})
    df= df.set_index("Sample")
    df = clean_genexpr_data(df)
    X, y = create_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    return X_train, X_test, y_train, y_test

