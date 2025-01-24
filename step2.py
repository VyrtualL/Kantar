import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import xgboost as xgb

df = pd.read_csv('EPITA_KANTAR_TP1/fic_epita_kantar_codes.csv', sep=';')
df = df.fillna(method='ffill')

behavior_vars = [
    'A11', 'A12', 'A13', 'A14', 'A4', 'A5', 'A5bis', 'A8_1_slice', 'A8_2_slice', 'A8_3_slice',
    'A8_4_slice', 'B1_1_slice', 'B1_2_slice', 'B2_1_slice', 'B2_2_slice', 'B3', 'B4', 'B6',
    'C1_1_slice', 'C1_2_slice', 'C1_3_slice', 'C1_4_slice', 'C1_5_slice', 'C1_6_slice',
    'C1_7_slice', 'C1_8_slice', 'C1_9_slice'
]

opinion_vars = [
    'A9_1_slice', 'A9_2_slice', 'A9_3_slice', 'A9_4_slice', 'A9_5_slice', 'A9_6_slice',
    'A9_7_slice', 'A9_8_slice', 'A9_9_slice', 'A9_10_slice', 'A9_11_slice', 'A9_12_slice',
    'A9_13_slice', 'A9_14_slice', 'A9_15_slice', 'A9_16_slice', 'A10_1_slice', 'A10_2_slice',
    'A10_3_slice', 'A10_4_slice', 'A10_5_slice', 'A10_6_slice', 'A10_7_slice', 'A10_8_slice',
    'A11_1_slice', 'A11_2_slice', 'A11_3_slice', 'A11_4_slice', 'A11_5_slice', 'A11_6_slice',
    'A11_7_slice', 'A11_8_slice', 'A11_9_slice', 'A11_10_slice', 'A11_11_slice', 'A11_12_slice',
    'A11_13_slice'
]


def prepare_data(df, variables):
    df_clean = df[variables].copy()
    df_clean = df_clean.replace('', np.nan)
    df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.fillna(df_clean.mean())

    return df_clean


def analyze_feature_importance(df, all_features, analysis_name, n_clusters=6):
    df_clean = prepare_data(df, all_features)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean)

    # Clustering
    kmeans_full = KMeans(n_clusters=n_clusters, random_state=42)
    reference_labels = kmeans_full.fit_predict(data_scaled)

    results = []
    feature_scores = []

    print(f"\nAnalyse des caractéristiques pour {analysis_name}...")

    for feature in all_features:
        X = data_scaled[:, [all_features.index(feature)]]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        score = adjusted_rand_score(reference_labels, labels)
        feature_scores.append({'feature': feature, 'importance': score})

    feature_importance = pd.DataFrame(feature_scores).sort_values('importance', ascending=False)

    for k in tqdm(range(1, len(all_features) + 1)):
        top_k_features = feature_importance.head(k)['feature'].tolist()
        feature_indices = [all_features.index(f) for f in top_k_features]
        data_subset = data_scaled[:, feature_indices]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_subset)

        silhouette = silhouette_score(data_subset, labels)
        rand_score = adjusted_rand_score(reference_labels, labels)

        results.append({
            'n_features': k,
            'features': top_k_features,
            'silhouette': silhouette,
            'rand_score': rand_score
        })

    return pd.DataFrame(results), feature_importance


def plot_comprehensive_results(results, feature_importance, analysis_name):
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Évolution des scores
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(results['n_features'], results['silhouette'], 'b-', label='Score Silhouette')
    ax1.plot(results['n_features'], results['rand_score'], 'r-', label='Score Rand Ajusté')
    ax1.set_xlabel('Nombre de variables')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Évolution de la performance du clustering - {analysis_name}')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Top 15 variables importantes
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    top_features = feature_importance.head(15)
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax2)
    ax2.set_title('15 variables les plus importantes')
    ax2.set_xlabel('Score d\'importance')

    # Plot 3: Distribution des scores d'importance
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    sns.histplot(data=feature_importance, x='importance', bins=20, ax=ax3)
    ax3.set_title('Distribution des scores d\'importance')
    ax3.set_xlabel('Score d\'importance')
    ax3.set_ylabel('Nombre de variables')

    plt.tight_layout()
    plt.show()

    optimal_point = results.loc[results['rand_score'].idxmax()]
    print(f"\nRésultats optimaux pour {analysis_name}:")
    print(f"Nombre optimal de variables: {optimal_point['n_features']}")
    print(f"Score de Rand ajusté maximal: {optimal_point['rand_score']:.3f}")
    print(f"Score de Silhouette correspondant: {optimal_point['silhouette']:.3f}")
    print("\nVariables recommandées:")
    for i, feature in enumerate(optimal_point['features'], 1):
        importance = feature_importance[feature_importance['feature'] == feature]['importance'].values[0]
        print(f"{i}. {feature} (importance: {importance:.3f})")


results_behavior, importance_behavior = analyze_feature_importance(
    df, behavior_vars, "Variables comportementales")
plot_comprehensive_results(results_behavior, importance_behavior, "Variables comportementales")

results_opinion, importance_opinion = analyze_feature_importance(
    df, opinion_vars, "Variables d'opinion")
plot_comprehensive_results(results_opinion, importance_opinion, "Variables d'opinion")

print("\nAnalyse comparative finale:")
print("\nVariables comportementales:")
print(f"Nombre total de variables: {len(behavior_vars)}")
print(f"Nombre optimal de variables: {results_behavior.loc[results_behavior['rand_score'].idxmax()]['n_features']}")
print(
    f"Réduction possible: {(1 - results_behavior.loc[results_behavior['rand_score'].idxmax()]['n_features'] / len(behavior_vars)) * 100:.1f}%")

print("\nVariables d'opinion:")
print(f"Nombre total de variables: {len(opinion_vars)}")
print(f"Nombre optimal de variables: {results_opinion.loc[results_opinion['rand_score'].idxmax()]['n_features']}")
print(
    f"Réduction possible: {(1 - results_opinion.loc[results_opinion['rand_score'].idxmax()]['n_features'] / len(opinion_vars)) * 100:.1f}%")


def xgboost_feature_importance(df, variables, n_clusters=6, n_iterations=10):
    data = prepare_data(df, variables)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    reference_labels = kmeans.fit_predict(data_scaled)

    feature_importances = []
    rand_scores = []
    silhouette_scores = []
    for _ in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            data_scaled, reference_labels, test_size=0.3, random_state=None
        )

        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': variables,
            'importance': importance
        }).sort_values('importance', ascending=False)

        feature_importances.append(feature_importance_df)

        results = []
        for k in range(1, len(variables) + 1):
            top_features = feature_importance_df.head(k)['feature'].tolist()
            feature_indices = [variables.index(f) for f in top_features]

            data_subset = data_scaled[:, feature_indices]

            # Clustering
            kmeans_subset = KMeans(n_clusters=n_clusters, random_state=42)
            labels_subset = kmeans_subset.fit_predict(data_subset)

            # Calcul des scores
            rand_score = adjusted_rand_score(reference_labels, labels_subset)
            silhouette = silhouette_score(data_subset, labels_subset)

            results.append({
                'n_features': k,
                'features': top_features,
                'rand_score': rand_score,
                'silhouette': silhouette
            })

        rand_scores.append(pd.DataFrame(results)['rand_score'].max())
        silhouette_scores.append(pd.DataFrame(results)['silhouette'].max())

    avg_feature_importance = feature_importances[0].copy()
    for df in feature_importances[1:]:
        avg_feature_importance['importance'] += df['importance']
    avg_feature_importance['importance'] /= len(feature_importances)

    results_df = pd.DataFrame(results)
    return (
        results_df,
        avg_feature_importance.sort_values('importance', ascending=False),
        {
            'avg_rand_score': np.mean(rand_scores),
            'std_rand_score': np.std(rand_scores),
            'avg_silhouette': np.mean(silhouette_scores),
            'std_silhouette': np.std(silhouette_scores)
        }
    )


def visualize_xgboost_results(results, feature_importance, stats, analysis_name):
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Évolution des scores
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(results['n_features'], results['silhouette'], 'b-', label='Score Silhouette')
    ax1.plot(results['n_features'], results['rand_score'], 'r-', label='Score Rand Ajusté')
    ax1.set_xlabel('Nombre de variables')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Évolution de la performance du clustering - {analysis_name} (XGBoost)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Top 15 variables importantes
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    top_features = feature_importance.head(15)
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax2)
    ax2.set_title('15 variables les plus importantes')
    ax2.set_xlabel('Score d\'importance')

    # Plot 3: Distribution des scores d'importance
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    sns.histplot(data=feature_importance, x='importance', bins=20, ax=ax3)
    ax3.set_title('Distribution des scores d\'importance')
    ax3.set_xlabel('Score d\'importance')
    ax3.set_ylabel('Nombre de variables')

    plt.tight_layout()
    plt.show()

    print(f"\nRésultats XGBoost pour {analysis_name}:")
    print(f"Score de Rand ajusté moyen: {stats['avg_rand_score']:.3f} ± {stats['std_rand_score']:.3f}")
    print(f"Score de Silhouette moyen: {stats['avg_silhouette']:.3f} ± {stats['std_silhouette']:.3f}")
    optimal_point = results.loc[results['rand_score'].idxmax()]
    print("\nVariables recommandées:")
    for i, feature in enumerate(optimal_point['features'], 1):
        importance = feature_importance[feature_importance['feature'] == feature]['importance'].values[0]
        print(f"{i}. {feature} (importance: {importance:.3f})")

    return optimal_point


print("Analyse XGBoost pour les variables comportementales:")
results_behavior_xgb, importance_behavior_xgb, stats_behavior_xgb = xgboost_feature_importance(
    df, behavior_vars)
optimal_behavior = visualize_xgboost_results(
    results_behavior_xgb, importance_behavior_xgb, stats_behavior_xgb, "Variables comportementales")

print("\nAnalyse XGBoost pour les variables d'opinion:")
results_opinion_xgb, importance_opinion_xgb, stats_opinion_xgb = xgboost_feature_importance(
    df, opinion_vars)
optimal_opinion = visualize_xgboost_results(
    results_opinion_xgb, importance_opinion_xgb, stats_opinion_xgb, "Variables d'opinion")


def compare_feature_sets(old_importance, xgboost_importance, name):
    print(f"\nComparaison des méthodes pour {name}:")

    old_top20 = set(old_importance.head(20)['feature'])
    xgb_top20 = set(xgboost_importance.head(20)['feature'])

    common_features = old_top20.intersection(xgb_top20)

    print("Top 20 variables communes:")
    for feature in common_features:
        old_rank = list(old_importance['feature']).index(feature)
        xgb_rank = list(xgboost_importance['feature']).index(feature)
        print(f"- {feature} (Ancien rang: {old_rank + 1}, XGBoost rang: {xgb_rank + 1})")

    print(f"\nNombre de variables communes: {len(common_features)}/20")
    print(f"Pourcentage de concordance: {len(common_features) / 20 * 100:.1f}%")

compare_feature_sets(
    importance_behavior,
    importance_behavior_xgb,
    "Variables comportementales"
)

compare_feature_sets(
    importance_opinion,
    importance_opinion_xgb,
    "Variables d'opinion"
)