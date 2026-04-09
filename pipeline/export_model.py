import joblib


def save_model(model):

    file_path = "trained_model.pkl"

    joblib.dump(
        model,
        file_path
    )

    return file_path