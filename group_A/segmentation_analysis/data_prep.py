import pandas as pd

def load_data(file_path):
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)

def clean_data(df, cols):
    """Perform data cleaning and preprocessing."""
    df = df[cols]
    return df