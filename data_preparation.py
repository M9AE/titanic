import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


DECKS_ENCODING = "TABCDEFGHI"
NAMES_DF = pd.read_csv("./data/names.csv")


def transform_cabin_deck(x):
    if pd.isna(x):
        return np.nan
    decks = [
        DECKS_ENCODING.index(c[0])
        for c in x.split(" ")
    ]
    return np.mean(decks)


def extract_cabin_bord(x):
    if pd.isna(x):
        return np.nan
    boards = [
        int(c[1:])%2==0 if len(c[1:]) > 0 else np.nan
        for c in x.split(" ")
    ]
    return np.median(boards)


def split_last(L, s):
    if isinstance(s, str):
        return L[:-1] + L[-1].split(s)
    elif isinstance(s, list):
        for s_ in s:
            if s_ in L[-1]:
                return L[:-1] + L[-1].split(s_)
    return L


def split_name(x):
    x = x.split(",")
    x = split_last(x, ".")
    x = split_last(x, ["(", "\""])
    return x


def apply_one_hot_encoding(df, col, encoder=None):
    subset = df[[col]]

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(subset)

    sex_columns = [col + "_" + c for c in encoder.categories_[0]]
    df[sex_columns[:-1]] = encoder.transform(subset).toarray()[:, :-1]
    df.drop(columns=[col], inplace=True)
    return df, encoder


def apply_knn_imputer(df, min_max_scaler=None, knn_imputer=None):
    if min_max_scaler is None:
        min_max_scaler = MinMaxScaler()
        df_min_max_scaled = min_max_scaler.fit_transform(df)
    else:
        df_min_max_scaled = min_max_scaler.transform(df)
    
    if knn_imputer is None:
        knn_imputer = KNNImputer(
            n_neighbors=3,
            weights="uniform",
            metric="nan_euclidean"
        )
        data_imputed = knn_imputer.fit_transform(df_min_max_scaled)
    else:
        data_imputed = knn_imputer.transform(df_min_max_scaled)

    descaled_data_imputed = min_max_scaler.inverse_transform(data_imputed)
    descaled_imputed_df = pd.DataFrame(
        descaled_data_imputed,
        columns=min_max_scaler.feature_names_in_,
    )
    return descaled_imputed_df, min_max_scaler, knn_imputer


def prepare_df(df):
    # 1. Drops
    prep_df = df.copy(deep=True).drop(columns=["Ticket"])

    # 2. One-hot encoding
    prep_df, sex_scaler = apply_one_hot_encoding(prep_df, "Sex")
    prep_df, embarked_scaler = apply_one_hot_encoding(prep_df, "Embarked")

    # 3. Transformation log
    prep_df["SibSp"] = prep_df["SibSp"].apply(lambda x: np.log(x+1))
    prep_df["Parch"] = prep_df["Parch"].apply(lambda x: np.log(x+1))
    prep_df["Fare"] = prep_df["Fare"].apply(lambda x: np.log(x+1))

    # 4. kNN imputer: Age
    prep_df, min_max_scaler, knn_imputer = apply_knn_imputer(prep_df.drop(columns=["Cabin", "Name"]))

    # 5. Cabin transformation
    prep_df["nb_cabins"] = df["Cabin"].str.count(" ") + 1
    prep_df["board"] = df["Cabin"].apply(extract_cabin_bord)
    prep_df["deck"] = df["Cabin"].apply(transform_cabin_deck)

    # 6. Name transformation
    name_df = df["Name"].apply(split_name).apply(pd.Series)
    prep_df["Zones"] = (
        pd.Series(name_df[0], name="last_name")
        .to_frame()
        .merge(NAMES_DF, how="left", left_on="last_name", right_on="Name")
        .loc[:, "Zone"]
    )
    prep_df["Title"] = name_df[1]
    prep_df, zones_scaling = apply_one_hot_encoding(prep_df, "Zones")
    prep_df, title_scaling = apply_one_hot_encoding(prep_df, "Title")

    preparation_model = {
        "sex_scaler": sex_scaler,
        "embarked_scaler": embarked_scaler,
        "min_max_scaler": min_max_scaler,
        "knn_imputer": knn_imputer,
        "zones_scaling": zones_scaling,
        "title_scaling": title_scaling,
    }
    return prep_df, preparation_model


def transform_df(df, preparation_model):
    # 1. Drops
    prep_df = df.copy(deep=True).drop(columns=["Ticket"])

    # 2. One-hot encoding
    prep_df, _ = apply_one_hot_encoding(prep_df, "Sex", preparation_model["sex_scaler"])
    prep_df, _ = apply_one_hot_encoding(prep_df, "Embarked", preparation_model["embarked_scaler"])

    # 3. Transformation log
    prep_df["SibSp"] = prep_df["SibSp"].apply(lambda x: np.log(x+1))
    prep_df["Parch"] = prep_df["Parch"].apply(lambda x: np.log(x+1))
    prep_df["Fare"] = prep_df["Fare"].apply(lambda x: np.log(x+1))

    # 4. kNN imputer
    prep_df, _, _ = apply_knn_imputer(
        prep_df.drop(columns=["Cabin", "Name"]),
        min_max_scaler=preparation_model["min_max_scaler"],
        knn_imputer=preparation_model["knn_imputer"],
    )

    # 5. Cabin transformation
    prep_df["nb_cabins"] = df["Cabin"].str.count(" ") + 1
    prep_df["board"] = df["Cabin"].apply(extract_cabin_bord)
    prep_df["deck"] = df["Cabin"].apply(transform_cabin_deck)

    # 6. Name transformation
    name_df = df["Name"].apply(split_name).apply(pd.Series)
    prep_df["Zones"] = (
        pd.Series(name_df[0], name="last_name")
        .to_frame()
        .merge(NAMES_DF, how="left", left_on="last_name", right_on="Name")
        .loc[:, "Zone"]
    )
    prep_df["Title"] = name_df[1]
    prep_df, _ = apply_one_hot_encoding(prep_df, "Zones", preparation_model["zones_scaling"])
    prep_df, _ = apply_one_hot_encoding(prep_df, "Title", preparation_model["title_scaling"])
    return prep_df
