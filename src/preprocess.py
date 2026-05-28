import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    return df.drop(columns=columns_to_drop)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
    df['tool_wear_torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
    return df

def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df = df.copy()
    df['Type'] = df['Type'].map(type_mapping)
    return df

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']
    return X, y

def split_train_test(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def preprocess_pipeline(filepath: str):
    df = load_data(filepath)
    df = drop_irrelevant_columns(df)
    df = engineer_features(df)
    df = encode_type(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

