import pandas as pd

df = pd.read_csv("EPITA_KANTAR_TP1/fic_epita_kantar_codes.csv", sep=";")
#print(df)

#first_cluster = df.loc[:, ["A9_1_slice", "A9_2_slice", "A9_3_slice", "A9_4_slice", "A9_5_slice", "A9_6_slice", "A9_7_slice", "A9_8_slice", "A9_9_slice", "A9_10_slice", "A9_11_slice", "A9_12_slice", "A9_13_slice", "A9_14_slice", "A9_15_slice", "A9_16_slice", "A10_1_slice", "A10_2_slice", "A10_3_slice", "A10_4_slice", "A10_5_slice", "A10_6_slice", "A10_7_slice", "A10_8_slice", "A10", "A11_1_slice", "A11_2_slice", "A11_3_slice", "A11_4_slice", "A11_5_slice", "A11_6_slice", "A11_7_slice", "A11_8_slice", "A11_9_slice", "A11_10_slice", "A11_11_slice", "A11_12_slice", "A11_13_slice", "A11"]]
#print(first_cluster)

p = df.loc[:, ["A9_1_slice", "A9_2_slice", "A9_3_slice", "A9_4_slice", "A9_5_slice", "A9_6_slice", "A9_7_slice", "A9_8_slice", "A9_9_slice", "A9_10_slice", "A9_11_slice", "A9_12_slice", "A9_13_slice", "A9_14_slice", "A9_15_slice", "A9_16_slice", "A10_1_slice", "A10_2_slice", "A10_3_slice", "A10_4_slice", "A10_5_slice", "A10_6_slice", "A10_7_slice", "A10_8_slice", "A11", "A11_1_slice", "A11_2_slice", "A11_3_slice", "A11_4_slice", "A11_5_slice", "A11_6_slice", "A11_7_slice", "A11_8_slice", "A11_9_slice", "A11_10_slice", "A11_11_slice", "A11_12_slice", "A11_13_slice"]]
#print(p)

p['A9'] = p.iloc[:, 0:15].mean(axis=1)
p['A10'] = p.iloc[:, 16:23].mean(axis=1)
p['A11bis'] = p.iloc[:, 25:37].mean(axis=1)
print(p)

#X = list(zip(p['A9'],p['A10'], p["A11"]))
X = p.loc[:, ['A9', 'A10', 'A11bis']].values
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

WCSS = []
for i in range(1,11):
    model = KMeans(n_clusters = i,init = 'k-means++')
    model.fit(X)
    WCSS.append(model.inertia_)
#fig = plt.figure(figsize = (7,7))
#plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'red')
#plt.xticks(np.arange(11))
#plt.xlabel("Number of clusters")
#plt.ylabel("WCSS")
#plt.show()

"""
model = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(X)
sns.countplot(y_clusters)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y_clusters == 0,0],X[y_clusters == 0,1],X[y_clusters == 0,2], s = 40 , color = 'blue', label = "cluster 0")
ax.scatter(X[y_clusters == 1,0],X[y_clusters == 1,1],X[y_clusters == 1,2], s = 40 , color = 'orange', label = "cluster 1")
ax.scatter(X[y_clusters == 2,0],X[y_clusters == 2,1],X[y_clusters == 2,2], s = 40 , color = 'green', label = "cluster 2")
ax.scatter(X[y_clusters == 3,0],X[y_clusters == 3,1],X[y_clusters == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
ax.scatter(X[y_clusters == 4,0],X[y_clusters == 4,1],X[y_clusters == 4,2], s = 40 , color = 'purple', label = "cluster 4")
ax.set_xlabel('Age of a customer-->')
ax.set_ylabel('Anual Income-->')
ax.set_zlabel('Spending Score-->')
ax.legend()
plt.show()
"""

d = df.loc[:, ["A11", "A12", "A13", "A14", "A4", "A5", "A5bis", "A8_1_slice", "A8_2_slice", "A8_3_slice", "A8_4_slice", "B1_1_slice", "B1_2_slice", "B2_1_slice", "B2_2_slice", "B3", "B4", "B6", "C1_1_slice", 'C1_2_slice', 'C1_3_slice', 'C1_4_slice', 'C1_5_slice', 'C1_6_slice', 'C1_7_slice', 'C1_8_slice', 'C1_9_slice']]
d["A8"] = d.iloc[:, 7:10].mean(axis=1)
d["B1"] = d.iloc[:, 11:12].mean(axis=1)
d["B2"] = d.iloc[:, 13:14].mean(axis=1)
d["C1"] = d.iloc[:, 18:26].mean(axis=1)
d["A5ter"] = d.iloc[:, 5:6].mean(axis=1)
d["A"] = d.iloc[:, [0, 1, 2, 3, 4, 5, 27, 31]].mean(axis=1)
d["B"] = d.iloc[:, [28, 29, 15, 16, 17]].mean(axis=1)
d["C"] = d.iloc[:, [30]].mean(axis=1)

x = d.loc[:, ["A", "B", "C"]].values


WCSS = []
for i in range(1,11):
    model = KMeans(n_clusters = i,init = 'k-means++')
    model.fit(x)
    WCSS.append(model.inertia_)
#fig = plt.figure(figsize = (7,7))
#plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'red')
#plt.xticks(np.arange(11))
#plt.xlabel("Number of clusters")
#plt.ylabel("WCSS")
#plt.show()

model = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(x)
sns.countplot(y_clusters)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],x[y_clusters == 0,2], s = 40 , color = 'blue', label = "cluster 0")
ax.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],x[y_clusters == 1,2], s = 40 , color = 'orange', label = "cluster 1")
ax.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],x[y_clusters == 2,2], s = 40 , color = 'green', label = "cluster 2")
ax.scatter(x[y_clusters == 3,0],x[y_clusters == 3,1],x[y_clusters == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
ax.scatter(x[y_clusters == 4,0],x[y_clusters == 4,1],x[y_clusters == 4,2], s = 40 , color = 'purple', label = "cluster 4")
ax.set_xlabel('Age of a customer-->')
ax.set_ylabel('Anual Income-->')
ax.set_zlabel('Spending Score-->')
ax.legend()
plt.show()