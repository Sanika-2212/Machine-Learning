import pandas as pd
import numpy as np
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#Step 1:- Load the data

#-------------------------------------------------------------------------------------
# Function Name : Load Data
# Description : Load the data from CSV
# Input : Name of CSV File
# Output : Data frame
# Author : Sanika Ashok Misal
# Date : 16/08/2026
#-------------------------------------------------------------------------------------

def LoadData(filename):
    df = pd.read_csv(filename)

    print("Dataset Loaded Successfully")

    print(df.head())

    return df

#-------------------------------------------------------------------------------------
# Function Name : main
# Description : Entry point function
# Input : None
# Output : None
# Author : Sanika Ashok Misal
# Date : 16/08/2026
#-------------------------------------------------------------------------------------

def main():
    LoadData("MarvellousTitanicDataset.csv")

if __name__ == "__main__":
    main()