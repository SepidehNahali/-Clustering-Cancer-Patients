# -Clustering-Cancer-Patients
# Multimodal Cancer Clustering with Deep Learning

## Overview

This project explores various deep learning pipelines for multimodal cancer data integration using clinical and somatic mutation data from the MSK-IMPACT cohort.

## Approach

To address our research questions, we developed and compared several deep learning pipelines: Similarity Network Fusion (SNF), Autoencoders (AEs), Graph Neural Networks (GNNs), and Contrastive Learning (CL). For each pipeline, we extracted a patient representation, performed clustering, and assessed the quality and relevance of the resulting clusters through three approaches:

1. Visualization and internal clustering metrics  
2. Survival outcome analysis using Kaplan-Meier curves  
3. Analysis of clinical and genomic variables  

In addition, we applied self-supervised contrastive learning methods to fine-tune patient representations by leveraging augmented views of clinical and mutation data to improve clustering performance.
