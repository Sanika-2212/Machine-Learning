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

#Step 2:- Data Preprocessing(EDA)

#-------------------------------------------------------------------------------------
# Function Name : PreprocessData
# Description : It Performs Exploratory Data Analysis
# Input : Dataframe
# Output : Updated Data frame
# Author : Sanika Ashok Misal
# Date : 16/08/2026
#-------------------------------------------------------------------------------------

def PreprocessData(df):
    df = df.drop([
        "Passengerid",
        "zero",
        "name"
    ],
    errors = "ignore"
    )

    # Handle missing values

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Convert Categorical to Numerical data

    df = pd.get_dummies(
        df,
        columns = ["Embarked"],
        drop_first= True,
        dtype= int
    )

    print(df.head())

    print("Data Preprocessing Completed")

    return df

#Step 3:- Split Data

#-------------------------------------------------------------------------------------
# Function Name : SplitData
# Description : It Performs Splitting Activity
# Input : Dataframe
# Output : 4 subset for training and testing
# Author : Sanika Ashok Misal
# Date : 16/08/2026
#-------------------------------------------------------------------------------------

def SplitData(df):
    X = df. drop("Survived", axis = 1)
    Y = df["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size= 0.2,
        random_state= 42
    )

    print("Dataset Splitting Completed Successfully")

    return X_train, X_test, Y_train, Y_test

#-------------------------------------------------------------------------------------
# Function Name : main
# Description : Entry point function
# Input : None
# Output : None
# Author : Sanika Ashok Misal
# Date : 16/08/2026
#-------------------------------------------------------------------------------------

def main():

    # Step 1:-
    df = LoadData("MarvellousTitanicDataset.csv")

    # Step 2:-
    df = PreprocessData(df)

    # Step 3:-
    X_train, X_test, Y_train, Y_test = SplitData(df)

if __name__ == "__main__":
    main()