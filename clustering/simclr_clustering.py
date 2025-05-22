import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def simclr_loss(z_i, z_j, temperature=0.5):
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    logits = tf.matmul(z_i, z_j, transpose_b=True) / temperature
    labels = tf.range(tf.shape(z_i)[0])
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)

def create_encoder(input_dim):
    return Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32)
    ])

def run_simclr_clustering(X_fused_feature, n_clusters=5):
    encoder = create_encoder(X_fused_feature.shape[1])
    optimizer = tf.keras.optimizers.Adam()
    batch_size = 64
    X = X_fused_feature.astype(np.float32)

    for epoch in range(10):
        for i in range(0, X.shape[0], batch_size):
            x1 = X[i:i+batch_size]
            x2 = x1.copy()
            with tf.GradientTape() as tape:
                z1 = encoder(x1)
                z2 = encoder(x2)
                loss = simclr_loss(z1, z2)
            grads = tape.gradient(loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
        print(f"SimCLR Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

    embeddings = encoder(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, clusters)
    print("SimCLR Clustering Silhouette Score:", score)
    return clusters, score
