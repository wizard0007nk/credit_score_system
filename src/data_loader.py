import pandas as pd

def load_data(path="C:/Users/nisha/Desktop/credit_score_system/data/raw/cs-training.csv"):
    df = pd.read_csv(path)
    return df
