import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')



def split():

    df = pd.read_csv("./data/raw_data/raw.csv")

    X= df.drop(columns=['silica_concentrate','date'])
    y=df['silica_concentrate']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    X_train_path = "./data/processed_data/X_train.csv"
    X_test_path = "./data/processed_data/X_test.csv"
    y_train_path = "./data/processed_data/y_train.csv"
    y_test_path = "./data/processed_data/y_test.csv"
    
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

if __name__ == "__main__":
    split()