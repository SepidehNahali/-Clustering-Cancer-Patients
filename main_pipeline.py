from data_loader import load_clinical_data, load_mutation_data
from fusion_methods import run_snf, build_autoencoder
from clustering import perform_clustering
from survival_analysis import plot_km_curves

# Load data
clinical = load_clinical_data("data/clinical.csv")
mutation = load_mutation_data("data/mutation.csv")

# SNF
fused = run_snf(clinical, mutation)
labels, score = perform_clustering(fused)
print("SNF Silhouette Score:", score)

# AE example
from sklearn.decomposition import PCA
ae_model, encoder = build_autoencoder(clinical.shape[1])
# Add training here...
# embeddings = encoder.predict(clinical)

# Plot KM for SNF
import pandas as pd
clinical_df = pd.read_csv("data/clinical.csv", index_col=0)
plot_km_curves(clinical_df, labels)
