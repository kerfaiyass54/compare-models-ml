from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score
)


def evaluate(task, model, X_test, y_test):

    predictions = model.predict(X_test)

    if task == "Regression":

        return {

            "MAE":
            mean_absolute_error(
                y_test,
                predictions
            ),

            "R2":
            r2_score(
                y_test,
                predictions
            )
        }

    return {

        "Accuracy":
        accuracy_score(
            y_test,
            predictions
        ),

        "F1":
        f1_score(
            y_test,
            predictions,
            average="weighted"
        )
    }