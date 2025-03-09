import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    """Splits data into features and labels, normalizes features."""
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("Preprocessing script is ready!")