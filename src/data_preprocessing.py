import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess():
    df = pd.read_csv("data/raw/crop_data.csv")

    os.makedirs("data/processed", exist_ok=True)

    imputer = SimpleImputer(strategy="mean")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = imputer.fit_transform(df[num_cols])

    le_dict = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    df.to_csv("data/processed/clean_data.csv", index=False)

    return df, le_dict