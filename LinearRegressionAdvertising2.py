import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def MarvellousRegression(DataPath):
    Border = "-" * 70

    #Step 1 :- Load the data
    print(Border)
    print("Step 1 :- Load the data")
    print(Border)

    df = pd.read_csv(DataPath)

    print(df.head())

    #Step 2 :- Remove Unwanted Columns from the dataset

    print(Border)
    print("Step 2 :- Remove Unwanted Columns from the dataset")
    print(Border)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(df.head())

def main():
    MarvellousRegression("Advertising.csv")

if __name__ == "__main__":
    main()