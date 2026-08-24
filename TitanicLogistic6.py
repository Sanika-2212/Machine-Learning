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

#Step 4:- Train the Model

#-------------------------------------------------------------------------------------
# Function Name : TrainModel
# Description : It Performs Model Training
# Input : Training features and labels
# Output : Trained Model
# Author : Sanika Ashok Misal
# Date : 16/08/2026
#-------------------------------------------------------------------------------------

def TrainModel(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)

    model = model.fit(X_train, Y_train)

    print("Model trained successfully")

    return model

#Step 5:- Evaluate the Model

#-------------------------------------------------------------------------------------
# Function Name : EvaluateModel
# Description : It Performs Model Testing
# Input : model, testing data( features, labels)
# Output : None
# Author : Sanika Ashok Misal
# Date : 16/08/2026
#-------------------------------------------------------------------------------------

def EvaluateModel(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy is :",accuracy)

    print(confusion_matrix(Y_test,Y_pred))

#Step 6:- Preserve the model

#-------------------------------------------------------------------------------------
# Function Name : PreserveModel
# Description : It Performs Model Preservation into .pkl file
# Input : model
# Output : None
# Author : Sanika Ashok Misal
# Date : 16/08/2026
#-------------------------------------------------------------------------------------

def PreserveModel(model,filename):
    joblib.dump(model,filename)

    print("Model Preserved with name :", filename)

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

    # Step 4:-
    model = TrainModel(X_train, Y_train)

    # Step 5:-
    EvaluateModel (model,X_test, Y_test)

    # Step 6:-
    PreserveModel(model,"MarvellousTitanic.pkl")

if __name__ == "__main__":
    main()
