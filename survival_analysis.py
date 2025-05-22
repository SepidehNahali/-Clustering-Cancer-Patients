import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def plot_km_curves(df, labels, time_col='time', event_col='event'):
    kmf = KaplanMeierFitter()
    for cluster in set(labels):
        ix = labels == cluster
        kmf.fit(df[time_col][ix], df[event_col][ix], label=f"Cluster {cluster}")
        kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curves")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.show()
