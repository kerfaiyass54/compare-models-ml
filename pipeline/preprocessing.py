import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(df):

    df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col].fillna(
                df[col].mode()[0],
                inplace=True
            )

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(
                df[col]
            )

        else:

            df[col].fillna(
                df[col].median(),
                inplace=True
            )

    return df