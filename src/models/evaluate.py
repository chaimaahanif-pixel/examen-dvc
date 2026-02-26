import pandas as pd
import joblib
import json
from sklearn.metrics import mean_absolute_error,mean_squared_error, root_mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def normalize():

    X_test = pd.read_csv("./data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("./data/processed_data/y_test.csv")
    model = joblib.load("./models/model.pkl")
    
    y_pred = model.predict(X_test)
    
    df_prediction = pd.DataFrame({
    'Real': y_test.values.ravel(),
    'Predicted': y_pred.ravel()
    })

    
    df_prediction.to_csv('./data/processed_data/y_pred.csv', index=False)

    mae = mean_absolute_error(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred) 
    rmse = root_mean_squared_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred) 
    
    with open("./metrics/scores.json", 'w') as f:
        json.dump({'R2': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse }, f)


if __name__ == "__main__":
    normalize()