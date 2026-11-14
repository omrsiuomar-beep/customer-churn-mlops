import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import CATEGORICAL, DATA_PATH, FEATURES, SEED, TARGET, TEST_SIZE


def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    df = df.copy()

    # TotalCharges comes in as text because new customers (tenure 0) have a blank value.
    # They haven't been billed yet, so 0 is the honest value, not the median.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()

    return df.drop(columns=["customerID"], errors="ignore")


def encode_target(y):
    encoder = LabelEncoder().fit(["No", "Yes"])
    return pd.Series(encoder.transform(y), index=y.index, name=TARGET)


def load_data(path=DATA_PATH):
    # a handful of rows are exact copies once the ID is gone, they would leak across the split
    return clean(load_raw(path)).drop_duplicates().reset_index(drop=True)


def split_data(df):
    X = df[FEATURES]
    y = encode_target(df[TARGET])
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
