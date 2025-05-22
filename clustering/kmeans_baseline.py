from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans_baseline(X_fused_feature, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100, algorithm='elkan')
    clusters = kmeans.fit_predict(X_fused_feature)
    score = silhouette_score(X_fused_feature, clusters)
    print("K-Means Silhouette Score:", score)
    return clusters, score
