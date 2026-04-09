from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)


def get_models(task):

    if task == "Regression":

        return {

            "Linear Regression":
            LinearRegression(),

            "Decision Tree":
            DecisionTreeRegressor(),

            "Random Forest":
            RandomForestRegressor(),

            "Gradient Boosting":
            GradientBoostingRegressor()

        }

    return {

        "Logistic Regression":
        LogisticRegression(max_iter=500),

        "Decision Tree":
        DecisionTreeClassifier(),

        "Random Forest":
        RandomForestClassifier(),

        "Gradient Boosting":
        GradientBoostingClassifier()

    }