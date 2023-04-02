from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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



def load_test_data_for_evaluation():
    df = load_data(configs["data_path"])
    df = preprocess_data(df)
    X, Y = create_X_y_from_gen_df(df, False, configs.get("label"))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)
    if configs["scale"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_test,y_test