import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_clinical_data(path):
    clinical = pd.read_csv(path, index_col=0)
    clinical = clinical.fillna(clinical.median(numeric_only=True))
    return StandardScaler().fit_transform(clinical)

def load_mutation_data(path):
    mutation = pd.read_csv(path, index_col=0)
    mutation = mutation.fillna(0)
    return mutation.values
