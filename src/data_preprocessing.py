import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath: str):
    original = pd.read_csv(filepath)
    data = original.copy()
    
    data = data.dropna()
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    return data

def split_data(data, target_col='Phising', test_size=0.2, random_state=42):
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
