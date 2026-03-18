from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np

class preprocessing:
    def __init__(self):
        pass

    def normalize_col(self,df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for col in df.columns:
            if col.lower() == "species":
                rename_map[col] = "Species"
            elif col.lower() == "originlocation":
                rename_map[col] = "OriginLocation"
        return df.rename(columns=rename_map)

    def get_preprocessed_df(self) -> pd.DataFrame:
        df = pd.read_csv("penguins.csv")
        df = self.normalize_col(df)
        null_columns = df.columns[df.isnull().any()]
        numeric_cols = df.select_dtypes(include="number").columns
        for col in null_columns:
            if col in numeric_cols:
                df[col] = df.groupby("Species")[col].transform(lambda x: x.fillna(x.mean()))
        le = LabelEncoder()
        df["OriginLocation"] = le.fit_transform(df["OriginLocation"])
        return df

    def split(self,data: pd.DataFrame):
        encoder = LabelEncoder()
        data["Species"] = encoder.fit_transform(data["Species"])

        # first_class = data[data["Species"] == -1]
        # second_class = data[data["Species"] == 1]
        # first_class= first_class.sample(frac=1).reset_index(drop=True)
        # second_class= second_class.sample(frac=1).reset_index(drop=True)
        # data = pd.concat([first_class, second_class], ignore_index=True)

        train_data = data.iloc[np.r_[0:30, 50:80, 100:130]]
        test_data  = data.iloc[np.r_[30:50, 80:100, 130:150]]

        sc = MinMaxScaler()
        train_data = sc.fit_transform(train_data)
        test_data  = sc.transform(test_data)

        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)

        X_train = train_data.iloc[:, 1:]
        y_train = train_data.iloc[:, 0:1]

        X_test = test_data.iloc[:, 1:]
        y_test = test_data.iloc[:, 0:1]

        # Keep original (unscaled) test coords for the scatter plot
        test_original = test_data.copy()

        return X_train, y_train, X_test, y_test, sc