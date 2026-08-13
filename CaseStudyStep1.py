import pandas as pd

Border = "-"*30

############################################
# Step 1 : Load the dataset
############################################

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