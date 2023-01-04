import pandas as pd
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
