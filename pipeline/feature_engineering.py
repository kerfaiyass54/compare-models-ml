def create_numeric_feature(df, col1, col2, operation):

    if operation == "add":
        df["new_feature"] = df[col1] + df[col2]

    elif operation == "mean":
        df["new_feature"] = (
            df[col1] + df[col2]
        ) / 2

    elif operation == "multiply":
        df["new_feature"] = df[col1] * df[col2]

    return df


def string_feature(df, column):

    df[column + "_length"] = df[column].str.len()

    return df