from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder
import pandas as pd
import numpy as np

class preprocessing:
    def __init__(self):
        pass

    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for col in df.columns:
            if col.lower() == "species":
                rename_map[col] = "Species"
            elif col.lower() == "originlocation":
                rename_map[col] = "OriginLocation"
        return df.rename(columns=rename_map)


    def get_preprocessed_df() -> pd.DataFrame:
        df = pd.read_csv("penguins.csv")
        df = normalize_columns(df)
        null_columns = df.columns[df.isnull().any()]
        numeric_cols = df.select_dtypes(include="number").columns
        for col in null_columns:
            if col in numeric_cols:
                df[col] = df.groupby("Species")[col].transform(lambda x: x.fillna(x.mean()))
        le = LabelEncoder()
        df["OriginLocation"] = le.fit_transform(df["OriginLocation"])
        return df


    def split(data: pd.DataFrame):

        encoder = LabelEncoder()
        data["Species"] = encoder.fit_transform(data["Species"])

        train_data = data.iloc[np.r_[0:30, 50:80, 100:130]].copy()
        test_data  = data.iloc[np.r_[30:50, 80:100, 130:150]].copy()

        X_train = train_data.drop(columns=["Species"])
        y_train = train_data[["Species"]]

        X_test = test_data.drop(columns=["Species"])
        y_test = test_data[["Species"]]

        sc = MinMaxScaler()
        X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)
        enc = OneHotEncoder(sparse_output=False)
        y_train_encoded = enc.fit_transform(y_train)
        y_test_encoded = enc.transform(y_test)
        y_train_encoded = pd.DataFrame(y_train_encoded)
        y_test_encoded = pd.DataFrame(y_test_encoded)
        return X_train, y_train_encoded, X_test, y_test_encoded, sc

