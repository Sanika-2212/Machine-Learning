import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    #Step 1 :- Load the Data

    df = pd.read_csv("Mall_Customers.csv")

    print("Dataset loaded with values")
    print(df.head())

    print("Missing Values:")
    print(df.isnull().sum())

    #Step 2 :- Feature Selection

    X = df[["AnnualIncome","SpendingScore"]]

    print("Selected features:")
    print(X.head())

    #Step 3 :- Scale the Data

    scalar = StandardScaler()

    X_scaled = scalar.fit_transform(X)

    print("Scaled Data:")
    print(X_scaled[:5])

    #Step 4 :- Elbow Method

    WCSS = []

    for k in range(1,11):
        model = KMeans(
            n_clusters = k,
            random_state = 42,
            n_init = 10,
        )

        model.fit(X_scaled)

        WCSS.append(model.inertia_)

    print("Values of WCSS :")

    for i in range(len(WCSS)):
        print(f"{i+1}:{WCSS[i]}")

if __name__ == "__main__":
    main()