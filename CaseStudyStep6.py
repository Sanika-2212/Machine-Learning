import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

Border = "-"*30

##############################################################
# Step 1 : Load the dataset
##############################################################

print(Border)
print("Step 1 : Load the dataset")
print(Border)

DataPath = "iris.csv"

df =pd.read_csv(DataPath)         #df = dataframe

print("Dataset loaded successfully")

print("Initial Entries from dataset are:")
print(df.head())        #first 5 entries

print("Last Entries from dataset are:")
print(df.tail())        #last 5 entries

##############################################################
# Step 2 : Exploratory Data Analysis (EDA)
##############################################################

print(Border)
print("Step 2 : Exploratory Data Analysis (EDA)")
print(Border)

print("Shape of dataset",df.shape)

print("Column Names:",list(df.columns))

print("Missing Values per column:")
print(df.isnull().sum())

print("Class Distribution (species count):")
print(df["species"].value_counts())

print("Statistical report of dataset:")
print(df.describe())

##############################################################
# Step 3 : Decide Independent and Dependent Variables
##############################################################

print(Border)
print("Step 3 : Decide Independent and Dependent Variables")
print(Border)

# X : Independent Variable / Features
# Y : Dependent Variable   / Labels

feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
    ]

X = df[feature_cols]
Y = df["species"]

print("X shape",X.shape)
print("Y shape",Y.shape)

##############################################################
# Step 4 : Visualization of dataset
##############################################################

print(Border)
print("Step 4 : Visualization of dataset")
print(Border)

#Scatter Plot
plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal length (cm)"], temp["petal width (cm)"], label = sp)

plt.title("Marvellous Iris Case Study")

plt.xlabel("petal length(cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid()
plt.show()

##############################################################
# Step 5 : Split the dataset for training and testing
##############################################################

print(Border)
print("Step 5 : Split the dataset for training and testing")
print(Border)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5,random_state=42)
print("Dataset Splitting Activity Done")

print("X:",X.shape)                 #(150,4)
print("Y:",Y.shape)                 #(150,)

print("X_train:",X_train.shape)     #(75,4)
print("X_test:",X_test.shape)       #(75,4)

print("Y_train:",Y_train.shape)     #(75,)
print("Y_test:",Y_test.shape)       #(75,)

##############################################################
# Step 6 : Build the model
##############################################################

print(Border)
print("Build the model")
print(Border)

model = DecisionTreeClassifier(max_depth = 5)

print("Model gets created successfully")
