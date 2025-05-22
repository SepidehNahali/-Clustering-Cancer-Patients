from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    return labels, score
