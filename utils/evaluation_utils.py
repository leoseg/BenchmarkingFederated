import wandb
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import configs
from data_utils import load_data, create_X_y_from_gen_df, preprocess_data
from models import get_model


def evaluate_model(weights,X_test,y_test):
    model = get_model(input_dim=configs["input_dim"], num_nodes=configs.get("num_nodes"),dropout_rate=configs.get("dropout_rate"), l1_v=configs.get("l1_v"), l2_v=configs.get("l2_v"))
    model.load_weights(weights)
    model.compile(loss= configs["loss"], optimizer=configs["optimizer"], metrics=configs["metrics"])
    score = model.evaluate(X_test, y_test, verbose = 0,return_dict=True)
    print(score)
    return score





def load_test_data_for_evaluation(repeat):
    data_path = configs["data_directory"]+ "unweighted_test_df_"+str(repeat)+".csv"
    df = load_data(data_path)
    df = preprocess_data(df)
    X_test, y_test = create_X_y_from_gen_df(df, False, configs.get("label"))
    if configs["scale"]:
        train_df = pd.read_csv(configs["data_directory"]+ "downsampled.csv")
        scaler = StandardScaler()
        train_df = preprocess_data(train_df)
        scaler.fit(train_df)
        X_test = scaler.transform(X_test)
    return X_test,y_test