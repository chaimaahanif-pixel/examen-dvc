import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


def train():

    X_train = pd.read_csv("./data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("./data/processed_data/y_train.csv")
    best_params = joblib.load("./models/best_params.pkl")
    
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "./models/model.pkl")
   

if __name__ == "__main__":
    train()