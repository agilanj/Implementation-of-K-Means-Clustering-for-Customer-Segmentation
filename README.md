# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Import data, inspect it, and handle missing values if any.
2. Determine optimal clusters: Use the Elbow Method to identify the number of clusters by plotting WCSS against cluster numbers.
3. Fit the K-Means model: Apply K-Means with the chosen number of clusters to the selected features.
4. Assign cluster labels to each data point.
5. Plot data points in a scatter plot, color-coded by cluster assignments for interpretation.

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: AGILAN J
RegisterNumber: 212224100002
```
```py
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
```
```py
data.head()
```
```py
data.info()
```
```py
data.isnull().sum()
```
```py
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++",n_init=10)
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
```
```py
km = KMeans(n_clusters=5, n_init=10)
km.fit(data.iloc[:, 3:])
y_pred = km.predict(data.iloc[:,3:])
y_pred
```
```py
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
```
```py
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster1")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster4")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster5")
plt.legend()
plt.title("Customer Segments")
```

## Output:

<img width="615" height="224" alt="image" src="https://github.com/user-attachments/assets/8302ff17-eb2e-4b08-b690-167d04a33e4a" />


<img width="619" height="268" alt="image" src="https://github.com/user-attachments/assets/e3d72cf6-f2d3-4044-aba0-a34cacd135be" />
<br>
<img width="328" height="141" alt="image" src="https://github.com/user-attachments/assets/97fb7ea0-9478-440e-8aa6-cc0a5410af7b" />

<img width="809" height="608" alt="image" src="https://github.com/user-attachments/assets/6c509019-7968-4eb3-8000-1644bf0c4e64" />

<img width="820" height="224" alt="image" src="https://github.com/user-attachments/assets/ae8f1876-9597-49e3-81c1-3aa2bea93f9a" />

<img width="768" height="591" alt="image" src="https://github.com/user-attachments/assets/551bd6f0-efcb-4384-b064-e00af4232926" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
