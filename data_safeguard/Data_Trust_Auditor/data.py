# data_trust_auditor/data_loader.py
import pandas as pd
import os

def load_dataset(filename: str):
    """
    Load a dataset from the data/ folder.
    Example: df = load_dataset("adult.csv")
    """
    data_dir = os.path.join(os.path.dirname(__file__), "adult")
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    return pd.read_csv(filepath)