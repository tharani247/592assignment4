import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch


def load_and_clean_data(filepath, target_column="CO(GT)", input_columns=None):
    df = pd.read_csv(filepath, sep=';', decimal=',', engine='python')

    # Drop last two unnamed columns
    df = df.drop(columns=df.columns[-2:])

    # Replace -200 with NaN
    df.replace(-200, np.nan, inplace=True)

    # Drop rows where target is NaN
    df = df.dropna(subset=[target_column])

    # Interpolate other columns
    df = df.infer_objects(copy=False)
    df.interpolate(method='linear', inplace=True)
    df = df.dropna()

    # Drop date/time columns
    df = df.drop(columns=['Date', 'Time'])

    if input_columns is not None:
        df = df[input_columns + [target_column]]

    return df


def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])
        y.append(data[i + seq_length, -1])  # Predict target at t+1
    return np.array(X), np.array(y)


def train_val_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, shuffle=False)


def prepare_data(filepath, target_column="CO(GT)", input_columns=None):
    df = load_and_clean_data(filepath, target_column, input_columns)
    data, scaler = scale_data(df)
    X, y = create_sequences(data)
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_val, y_val, scaler
