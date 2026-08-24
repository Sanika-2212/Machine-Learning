import pandas as pd
import joblib

def LoadModel(Filename):
    model = joblib.load(Filename)

    print("Model loaded successfully")

    print(model.feature_names_in_)

    return model

def PredictPassenger(model):
    print("Enter the information :")

    Pclass = int(input("Enter Pclass (1/2/3):"))
    Sex = int(input("Enter Sex (0-F /1-M):"))
    Age = int(input("Enter your Age :"))
    sibsp = int(input("Enter sibsp :"))
    Parch = int(input("Enter Parch :"))
    Fare = float(input("Enter Fare Details :"))
    Embarked = float(input("Enter Embarked (0/1/2):"))

    passenger = pd.DataFrame([{
        "Pclass" : Pclass,
        "Sex" : Sex,
        "Age" : Age,
        "sibsp" : sibsp,
        "Parch" : Parch,
        "Fare" : Fare,
        "Embarked_1.0" : 1 if Embarked == 1 else 0,
        "Embarked_1.0" : 1 if Embarked == 2 else 0,
    }])

    passenger = passenger[model.feature_names_in_]

    result = model.predict(passenger)

    print(result)

def main():
    model = LoadModel("MarvellousTitanic.pkl")

    PredictPassenger(model)

if __name__ == "__main__":
    main()