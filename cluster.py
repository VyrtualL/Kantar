import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
======================================
1 - Clustering
Une clusterisation des 5000 individus en utilisant les variables A9, A10, A11
Une clusterisation des 5000 individus en utilisant les variables A11, A12, A13, A14, A4, A5,...


Vous devrez justifier du choix de la méthode de clusterisation (ou des méthodes) 
utilisée(s) et expliquer les raisons du choix de la version de clustering 
choisie (nb de groupes). Calcul des variances intra groupes et inter groupes. (ratio)
Pour chacun des groupes finaux, un descriptif succinct devra être fourni.
"""

"""
#On recupere le CSV
"""
df = pd.read_csv("EPITA_KANTAR_TP1/fic_epita_kantar_codes.csv", sep=";")
#print(df)

#first_cluster = df.loc[:, ["A9_1_slice", "A9_2_slice", "A9_3_slice", "A9_4_slice", "A9_5_slice", "A9_6_slice", "A9_7_slice", "A9_8_slice", "A9_9_slice", "A9_10_slice", "A9_11_slice", "A9_12_slice", "A9_13_slice", "A9_14_slice", "A9_15_slice", "A9_16_slice", "A10_1_slice", "A10_2_slice", "A10_3_slice", "A10_4_slice", "A10_5_slice", "A10_6_slice", "A10_7_slice", "A10_8_slice", "A10", "A11_1_slice", "A11_2_slice", "A11_3_slice", "A11_4_slice", "A11_5_slice", "A11_6_slice", "A11_7_slice", "A11_8_slice", "A11_9_slice", "A11_10_slice", "A11_11_slice", "A11_12_slice", "A11_13_slice", "A11"]]
#print(first_cluster)

"""
#On recupere les colonnes qui nous intéresse

Colonne Weight ?

Dans les réponses A9, on constate des questions avec des réponses :
Tout à fait d'accord (0) -> Pas du tout d'accord (4)
Les questions sont sur l'interet de la personne et son investissement sur le jardinage
 et les espaces extérieurs.
 Mais une question (A9_5_slide) est formulé de telle sorte à ce que si 
  l'on réponds "Tout à fait d'accord", cela soit négatif, alors que c'est tout
  l'inverse dans les autres questions 
       - "Je préfère que les espaces extérieurs soient plutôt sauvages que très entretenus"
Il faut donc inverser ses valeurs (0 -> 4,...)



Dans les réponses A10, on constate aussi des questions avec des réponses :
Tout à fait d'accord (0) -> Pas du tout d'accord (4)
Ici, c'est des questions pour connaitre notre opinion sur les espaces extérieurs
Encore, une question (A10_6_slide) est formulé de telle sorte à ce que si 
  l'on réponds "Tout à fait d'accord", cela soit négatif, alors que c'est tout
  l'inverse dans les autres questions
       - "Les espaces extérieurs sont surtout des sources de contraintes"
Il faut donc inverser ses valeurs (0 -> 4,...)
On peut aussi se questionner sur cette question :
       - "Les espaces extérieurs valorisent les biens immobiliers" 
Mais nous n'allons pas la modifier



Dans les réponses A11, on constate aussi des questions avec des réponses :
Tout à fait d'accord (0) -> Pas du tout d'accord (4)
Ici, c'est des questions pour savoir a quoi nous sert un jardin et qu'est ce que
  cela nous permet d'en faire
Toutes les questions ont l'air bien posé


Maintenant ce que l'on peut faire, c'est reduire la dimensionnalité de chaque variable 
  en les regroupant.
A9, A10 et A11 aillant leurs propres thèmes (au niveau des questions), on peut regrouper
  les réponses en 1 dimensions pour chaque variable. Pour cela, on peut utiliser 
  une Analyse en Composantes Principales (PCA). Grâce au prétraitementt que l'on a pu faire avant,
  cela va nous permettre de bien différentier ceux qui on repondu négativement, "neutre" ou positivement,
  avec 1 seule valeur. 
  
  
Pour le cluster, nous avons le choix de 3 clusters différents :
K-means / Clustering Hiérarchique / DBSCAN

Commençons tout d'abord par un K-Means

"""

"""
On recupere nos données
"""
A9 = df.loc[:, ["A9_1_slice", "A9_2_slice", "A9_3_slice", "A9_4_slice", "A9_5_slice", "A9_6_slice", "A9_7_slice", "A9_8_slice", "A9_9_slice", "A9_10_slice", "A9_11_slice", "A9_12_slice", "A9_13_slice", "A9_14_slice", "A9_15_slice", "A9_16_slice"]]
A10 = df.loc[:, ["A10_1_slice", "A10_2_slice", "A10_3_slice", "A10_4_slice", "A10_5_slice", "A10_6_slice", "A10_7_slice", "A10_8_slice"]]
A11 = df.loc[:, ["A11_1_slice", "A11_2_slice", "A11_3_slice", "A11_4_slice", "A11_5_slice", "A11_6_slice", "A11_7_slice", "A11_8_slice", "A11_9_slice", "A11_10_slice", "A11_11_slice", "A11_12_slice", "A11_13_slice"]]

"""
On inverse les valeurs pour les questions "négatives"
"""
<<<<<<< HEAD
A9["A9_5_slie"] = A9["A9_5_slice"].map({1: 4, 2: 3, 3: 2, 4: 1})
=======
A9["A9_5_slice"] = A9["A9_5_slice"].map({1: 4, 2: 3, 3: 2, 4: 1})
>>>>>>> 784a86543a44be4ef217c3d2b9801e0cff499854
A10["A10_6_slice"] = A10["A10_6_slice"].map({1: 4, 2: 3, 3: 2, 4: 1})

"""
On fait le PCA pour réduire les dimensions pour A9, A10, A11
"""
scaler = StandardScaler()
A9_normalized = scaler.fit_transform(A9)
A10_normalized = scaler.fit_transform(A10)
A11_normalized = scaler.fit_transform(A11)

A9_pca = PCA(n_components=1)
A10_pca = PCA(n_components=1)
A11_pca = PCA(n_components=1)

A9_pca_result = A9_pca.fit_transform(A9_normalized)
A10_pca_result = A10_pca.fit_transform(A10_normalized)
A11_pca_result = A11_pca.fit_transform(A11_normalized)

A9_pca_df = pd.DataFrame(A9_pca_result, columns=["PCA1_value"])
A10_pca_df = pd.DataFrame(A10_pca_result, columns=["PCA2_value"])
A11_pca_df = pd.DataFrame(A11_pca_result, columns=["PCA3_value"])

"""
On a deux choix : 
- Soit concaténer les 3 dataframes pour faire 1 seule dimension pour le cluster
- Soit garder les 3 dataframes pour faire un cluster à 3 dimensions
Garder les 3 dimensions est préférable car préserve la richesse des données et leurs relations
Représentation en 3D

Nous choisirons d'abord 4 clusters pour les 4 choix de reponses disponible
"""

data_cluster = pd.concat([A9_pca_df, A10_pca_df, A11_pca_df], axis=1)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(data_cluster)

data_cluster_array = data_cluster.iloc[:, :3].values
linkage_matrix = linkage(data_cluster_array, method="ward")
hierarchical_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_cluster)

data_cluster['KMeans_Label'] = kmeans_labels
data_cluster['Hierarchical_Label'] = hierarchical_labels
data_cluster['DBSCAN_Label'] = dbscan_labels

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_cluster.iloc[:, 0], data_cluster.iloc[:, 1], data_cluster.iloc[:, 2],
           c=data_cluster['KMeans_Label'], cmap='viridis', s=50)
ax.set_title("K-Means Clustering")
ax.set_xlabel("PCA1 (A9)")
ax.set_ylabel("PCA2 (A10)")
ax.set_zlabel("PCA3 (A11)")
plt.show()

plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title("Clustering Hiérarchique")
plt.show()


#data_3d_array = data_cluster[["PCA1_value", "PCA2_value", "PCA3_value"]].values
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_cluster.iloc[:, 0], data_cluster.iloc[:, 1], data_cluster.iloc[:, 2],
           c=data_cluster['DBSCAN_Label'], cmap='viridis', s=50)
ax.set_title("DBSCAN Clustering")
ax.set_xlabel("PCA1 (A9)")
ax.set_ylabel("PCA2 (A10)")
ax.set_zlabel("PCA3 (A11)")
plt.show()


"""
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
"""