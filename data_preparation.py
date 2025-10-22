import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def apply_one_hot_encoding(df, col, encoder=None):
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore")

    subset = df[[col]]

    encoder.fit(subset)
    sex_columns = ["Sex_" + col for col in encoder.categories_[0]]

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
    prep_df = (
        df
        .copy(deep=True)
        .drop(columns=["Ticket", "Cabin", "Name"])
    )

    # 2. One-hot encoding
    prep_df, sex_scaler = apply_one_hot_encoding(prep_df, "Sex")
    prep_df, embarked_scaler = apply_one_hot_encoding(prep_df, "Embarked")

    # 3. Transformation log
    prep_df["SibSp"] = prep_df["SibSp"].apply(lambda x: np.log(x+1))
    prep_df["Parch"] = prep_df["Parch"].apply(lambda x: np.log(x+1))
    prep_df["Fare"] = prep_df["Fare"].apply(lambda x: np.log(x+1))

    # 4. kNN imputer
    prep_df, min_max_scaler, knn_imputer = apply_knn_imputer(prep_df)

    preparation_model = {
        "sex_scaler": sex_scaler,
        "embarked_scaler": embarked_scaler,
        "min_max_scaler": min_max_scaler,
        "knn_imputer": knn_imputer,
    }
    return prep_df, preparation_model


def transform_df(df, preparation_model):
    # 1. Drops
    prep_df = (
        df
        .copy(deep=True)
        .drop(columns=["Ticket", "Cabin", "Name"])
    )

    # 2. One-hot encoding
    prep_df, _ = apply_one_hot_encoding(prep_df, "Sex", preparation_model["sex_scaler"])
    prep_df, _ = apply_one_hot_encoding(prep_df, "Embarked", preparation_model["embarked_scaler"])

    # 3. Transformation log
    prep_df["SibSp"] = prep_df["SibSp"].apply(lambda x: np.log(x+1))
    prep_df["Parch"] = prep_df["Parch"].apply(lambda x: np.log(x+1))
    prep_df["Fare"] = prep_df["Fare"].apply(lambda x: np.log(x+1))

    # 4. kNN imputer
    prep_df, _, _ = apply_knn_imputer(
        prep_df,
        min_max_scaler=preparation_model["min_max_scaler"],
        knn_imputer=preparation_model["knn_imputer"],
    )
    return prep_df
