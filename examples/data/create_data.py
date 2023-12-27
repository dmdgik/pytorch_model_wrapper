import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_data_for_regression(n_objects, n_features):
    
    feature_columns = [f"feature_{i}" for i in range(n_features)]
    columns = feature_columns + ["target"]
    features = np.random.rand(n_objects, n_features)
    targets = np.random.rand(n_objects, 1)
    data = np.concatenate([features, targets], axis=1)
    df = pd.DataFrame(data=data, columns=columns)
    df_, df_test = train_test_split(df, test_size=0.1, shuffle=True)
    df_train, df_valid = train_test_split(df_, test_size=0.2, shuffle=True)
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_train["id"] = df_train.index
    df_valid["id"] = df_valid.index
    df_test["id"] = df_test.index
    df_train.to_csv("./data_regression_train.csv", index=False)
    df_valid.to_csv("./data_regression_valid.csv", index=False)
    df_test.to_csv("./data_regression_test.csv", index=False)
    
def create_data_for_classification(n_objects, n_features, n_classes):
    
    feature_columns = [f"feature_{i}" for i in range(n_features)]
    columns = feature_columns + ["target"]
    features = np.random.rand(n_objects, n_features)
    targets = np.random.randint(0, n_classes, (n_objects, 1))
    data = np.concatenate([features, targets], axis=1)
    df = pd.DataFrame(data=data, columns=columns)
    df_, df_test = train_test_split(df, test_size=0.1, shuffle=True)
    df_train, df_valid = train_test_split(df_, test_size=0.2, shuffle=True)
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_train["id"] = df_train.index
    df_valid["id"] = df_valid.index
    df_test["id"] = df_test.index
    df_train.to_csv("./data_classification_train.csv", index=False)
    df_valid.to_csv("./data_classification_valid.csv", index=False)
    df_test.to_csv("./data_classification_test.csv", index=False)


if __name__=="__main__":
    create_data_for_regression(3000, 10)
    create_data_for_classification(3000, 10, 3)