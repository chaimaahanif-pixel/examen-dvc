import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')


def gridsearch():

    X_train = pd.read_csv("./data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("./data/processed_data/y_train.csv")
    
    param_grid = {
    "n_estimators": [50, 200, 500],
    "max_depth": [None, 15, 30],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", 0.7],
    "bootstrap": [True]
}
    
    grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    joblib.dump(grid.best_params_, "./models/best_params.pkl")


if __name__ == "__main__":
    gridsearch()