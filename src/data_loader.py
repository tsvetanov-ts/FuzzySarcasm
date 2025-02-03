import pandas as pd
import os

def load_data(file_path):
    """
    Load a CSV file and return a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    df = pd.read_csv(file_path)
    return df

def load_train_test(train_path, test_path):
    """
    Load both the training and test datasets.
    """
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    return train_df, test_df