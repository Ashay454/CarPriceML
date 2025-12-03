import pandas as pd
import numpy as np
import datetime

def load_data(filepath):
    """Loads CSV data."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None

def clean_data(df):
    
    df = df.copy()
    
    if 'max_power' in df.columns:
        df['max_power'] = df['max_power'].astype(str).str.replace(' bhp', '', regex=False)
        df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

    if 'mileage' in df.columns:
        df['mileage'] = df['mileage'].astype(str).str.replace(' kmpl', '', regex=False)
        df['mileage'] = df['mileage'].astype(str).str.replace(' km/kg', '', regex=False)
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

    if 'engine' in df.columns:
        df['engine'] = df['engine'].astype(str).str.replace(' CC', '', regex=False)
        df['engine'] = pd.to_numeric(df['engine'], errors='coerce')

    df.dropna(inplace=True)
    
    return df

def feature_engineering(df):
    """
    Adds new features to help the model learn better.
    """
    df = df.copy()
    
    current_year = datetime.datetime.now().year
    if 'year' in df.columns:
        df['car_age'] = current_year - df['year']
        df.drop('year', axis=1, inplace=True)
        
    return df

def prepare_features(df):
    df = feature_engineering(df)
    
    X = df.drop(['selling_price', 'name'], axis=1, errors='ignore')
    y = df['selling_price']
    return X, y