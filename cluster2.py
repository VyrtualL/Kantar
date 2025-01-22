import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from factor_analyzer import FactorAnalyzer
import prince
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv('EPITA_KANTAR_TP1/fic_epita_kantar_codes.csv', sep=';')
df = df.fillna(method='ffill')

# 1. Analyse des corrélations
def analyze_correlations(data, variables):
    corr_matrix = data[variables].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Matrice de corrélations')
    plt.tight_layout()
    plt.show()


# 2. PCA et Analyse Factorielle
def perform_pca_analysis(data, variables):
    # Standardisation
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[variables])

    # PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)

    # Valeurs propres
    eigenvalues = pca.explained_variance_

    # Variance expliquée
    var_explained = pca.explained_variance_ratio_

    return pca, pca_result, eigenvalues, var_explained


# 3. Clustering Hiérarchique
def hierarchical_clustering(data, n_clusters):
    linkage_matrix = linkage(data, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('Dendrogramme')
    plt.show()

    hc = AgglomerativeClustering(n_clusters=n_clusters)
    return hc.fit_predict(data)


# 4. K-means Clustering
def kmeans_clustering(data, max_clusters=10):
    silhouette_scores = []
    ch_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
        ch_scores.append(calinski_harabasz_score(data, labels))

    return silhouette_scores, ch_scores


# 5. Description des clusters
def describe_clusters(data, labels, variables):
    df_with_clusters = data.copy()
    df_with_clusters['Cluster'] = labels

    cluster_descriptions = df_with_clusters.groupby('Cluster')[variables].mean()
    return cluster_descriptions


# 6. Golden Questions
def find_golden_questions(data, cluster_labels, variables):
    X = data[variables]
    y = cluster_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Utiliser un arbre de décision pour identifier les variables les plus importantes
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)

    feature_importance = pd.DataFrame({
        'feature': variables,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)

    return feature_importance


# Application sur les deux ensembles de variables
# Premier clustering (variables orange)
orange_vars = ['A9_1_slice', 'A10_1_slice', 'A11_1_slice']  # Ajustez selon vos besoins

# Deuxième clustering (variables vertes)
green_vars = ['A11', 'A12', 'A13', 'A14', 'A4', 'A5', 'A8_1_slice', 'A8_2_slice',
              'B1_1_slice', 'B1_2_slice', 'B2_1_slice', 'B2_2_slice', 'B3', 'B4', 'B6']


def create_persona(cluster_description):
    """
    Crée un persona basé sur les caractéristiques du cluster
    """
    persona = {
        "profil": {},
        "comportements": {},
        "attitudes": {},
        "besoins": {}
    }

    # Remplir avec les caractéristiques principales
    for var, value in cluster_description.items():
        if var.startswith('A'):
            persona["profil"][var] = value
        elif var.startswith('B'):
            persona["comportements"][var] = value
        elif var.startswith('C'):
            persona["attitudes"][var] = value

    return persona


def analyze_cluster_set(data, variables, n_clusters=5):
    # Standardisation
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[variables])

    # Analyse des corrélations
    analyze_correlations(data, variables)

    # PCA
    pca, pca_result, eigenvalues, var_explained = perform_pca_analysis(data, variables)

    # Clustering
    silhouette_scores, ch_scores = kmeans_clustering(data_scaled)

    # Sélection finale du nombre de clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)

    # Description des clusters
    cluster_desc = describe_clusters(data, cluster_labels, variables)

    # Golden Questions
    golden_questions = find_golden_questions(data, cluster_labels, variables)

    # Création des personas
    personas = {}
    for cluster in range(n_clusters):
        personas[f"Cluster_{cluster}"] = create_persona(cluster_desc.loc[cluster])

    return {
        'cluster_labels': cluster_labels,
        'cluster_descriptions': cluster_desc,
        'golden_questions': golden_questions,
        'personas': personas,
        'pca_results': {
            'eigenvalues': eigenvalues,
            'variance_explained': var_explained
        }
    }

cluster = analyze_cluster_set(df, variables=green_vars, n_clusters=5)
print(cluster)

print("== FIN DE CLAUDE ==")
print("== DEBUT DE CHATGPT ==")

import pandas as pd

# Charger les données
df = pd.read_csv("EPITA_KANTAR_TP1/fic_epita_kantar_codes.csv", sep=";")

# Sélection des colonnes
comportement_cols = ["A11", "A12", "A13", "A14", "A4", "A5", "A5bis", "A8_1_slice", "A8_2_slice", "A8_3_slice",
                     "A8_4_slice", "B1_1_slice", "B1_2_slice", "B2_1_slice", "B2_2_slice", "B3", "B4", "B6",
                     "C1_1_slice", "C1_2_slice", "C1_3_slice", "C1_4_slice", "C1_5_slice", "C1_6_slice",
                     "C1_7_slice", "C1_8_slice", "C1_9_slice"]
opinion_cols = ["A9_1_slice", "A9_2_slice", "A9_3_slice", "A9_4_slice", "A9_5_slice", "A9_6_slice", "A9_7_slice",
                "A9_8_slice", "A9_9_slice", "A9_10_slice", "A9_11_slice", "A9_12_slice", "A9_13_slice", "A9_14_slice",
                "A9_15_slice", "A9_16_slice", "A10_1_slice", "A10_2_slice", "A10_3_slice", "A10_4_slice",
                "A10_5_slice", "A10_6_slice", "A10_7_slice", "A10_8_slice", "A11_1_slice", "A11_2_slice",
                "A11_3_slice", "A11_4_slice", "A11_5_slice", "A11_6_slice", "A11_7_slice", "A11_8_slice",
                "A11_9_slice", "A11_10_slice", "A11_11_slice", "A11_12_slice", "A11_13_slice"]

df_comportement = df[comportement_cols]
df_opinion = df[opinion_cols]

import seaborn as sns
import matplotlib.pyplot as plt

# Statistiques descriptives
print(df_comportement.describe())
print(df_opinion.describe())

# Matrice de corrélation
sns.heatmap(df_comportement.corr(), annot=False, cmap="coolwarm")
plt.title("Corrélations des variables comportement")
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Normalisation des données
scaler = StandardScaler()
X_comportement = scaler.fit_transform(df_comportement.fillna(0))
X_opinion = scaler.fit_transform(df_opinion.fillna(0))

# PCA
pca = PCA(n_components=2)
X_pca_comportement = pca.fit_transform(X_comportement)
X_pca_opinion = pca.fit_transform(X_opinion)

print("Variance expliquée (comportement):", pca.explained_variance_ratio_)
print("Variance expliquée (opinion):", pca.explained_variance_ratio_)

# Visualisation
plt.scatter(X_pca_comportement[:, 0], X_pca_comportement[:, 1], alpha=0.5)
plt.title("PCA - Comportement")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Visualisation
plt.scatter(X_pca_opinion[:, 0], X_pca_opinion[:, 1], alpha=0.5)
plt.title("PCA - Opinion")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

# Dendrogramme
linked = linkage(X_comportement, method="ward")
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode="level", p=5)
plt.title("Dendrogramme - Comportement")
plt.show()

linked = linkage(X_opinion, method="ward")
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode="level", p=5)
plt.title("Dendrogramme - Opinion")
plt.show()

# K-means
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_comportement)
df["Cluster"] = kmeans.labels_

numeric_cols = df.select_dtypes(include=["number"]).columns
cluster_summary = df.groupby("Cluster")[numeric_cols].mean()
print(cluster_summary)

from scipy.stats import f_oneway

# ANOVA pour chaque colonne
anova_results = {col: f_oneway(*[df[df["Cluster"] == k][col] for k in df["Cluster"].unique()])
                 for col in comportement_cols + opinion_cols}

# Variables significatives
significant_vars = [k for k, v in anova_results.items() if v.pvalue < 0.05]
print("Questions d'or:", significant_vars)


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Fonction pour effectuer PCA et clustering
def plot_clusters_pca(data, n_clusters, title):
    # Normalisation des données
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.fillna(0))  # Remplissage des valeurs manquantes

    # PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_scaled)
    print(f"{title} - Variance expliquée par PC1 et PC2 :", pca.explained_variance_ratio_)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_scaled)

    # Visualisation
    plt.figure(figsize=(8, 6))
    for cluster in range(n_clusters):
        cluster_points = pca_data[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    plt.title(f"Clusters selon PCA - {title}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

# Étude comportement
plot_clusters_pca(df_comportement, n_clusters=6, title="Comportement")

# Étude opinion
plot_clusters_pca(df_opinion, n_clusters=6, title="Opinion")


def plot_3d_clusters(data, variables, n_clusters=5):
    # Standardisation
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[variables])

    # PCA pour réduction à 3 dimensions
    pca = PCA(n_components=3)
    components = pca.fit_transform(data_scaled)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Création du graphique 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Palette de couleurs
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))

    # Tracer chaque cluster
    for i in range(n_clusters):
        mask = clusters == i
        ax.scatter(components[mask, 0],
                   components[mask, 1],
                   components[mask, 2],
                   c=[colors[i]],
                   label=f'Cluster {i + 1}')

    # Labels et titre
    ax.set_xlabel('Première composante')
    ax.set_ylabel('Deuxième composante')
    ax.set_zlabel('Troisième composante')
    ax.set_title('Visualisation 3D des clusters')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Afficher la variance expliquée
    print(f"Variance expliquée par les 3 composantes: {sum(pca.explained_variance_ratio_) * 100:.2f}%")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"Composante {i + 1}: {var * 100:.2f}%")


# Pour les variables orange
orange_vars = ['A9_1_slice', 'A10_1_slice', 'A11_1_slice']
plot_3d_clusters(df, orange_vars)

# Pour les variables vertes
green_vars = ['A11', 'A12', 'A13', 'A14', 'A4', 'A5', 'A8_1_slice', 'A8_2_slice',
              'B1_1_slice', 'B1_2_slice', 'B2_1_slice', 'B2_2_slice', 'B3', 'B4', 'B6']
plot_3d_clusters(df, green_vars)


# Version interactive avec rotation
def plot_3d_clusters_interactive(data, variables, n_clusters=5):
    from matplotlib.animation import FuncAnimation

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[variables])

    pca = PCA(n_components=3)
    components = pca.fit_transform(data_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
    scatters = []

    for i in range(n_clusters):
        mask = clusters == i
        scatter = ax.scatter(components[mask, 0],
                             components[mask, 1],
                             components[mask, 2],
                             c=[colors[i]],
                             label=f'Cluster {i + 1}')
        scatters.append(scatter)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.legend()

    def rotate(frame):
        ax.view_init(elev=30., azim=frame)
        return scatters

    ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2),
                        interval=50, blit=True)
    plt.show()


# Version interactive
plot_3d_clusters_interactive(df, orange_vars)
plot_3d_clusters_interactive(df, green_vars)