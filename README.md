# Multimodal Cancer Clustering with Deep Learning

This project investigates advanced methods for clustering cancer patients using multimodal data, including clinical information and somatic mutation profiles from the MSK-IMPACT dataset. The goal is to discover meaningful patient subgroups associated with therapy-induced clonal hematopoiesis and survival outcomes.

  📌 Objective

To compare the effectiveness of different multimodal integration techniques and clustering strategies in identifying clinically and genomically meaningful cancer patient subgroups.

  🧠 Methods Compared

We developed and evaluated the following deep learning pipelines:

- **Similarity Network Fusion (SNF)**
- **Autoencoders (AEs)**
- **Graph Neural Networks (GNNs)**
- **Contrastive Learning (CL)**

Each method is used to extract patient representations from clinical and mutation data.

  🔍 Evaluation Strategy

To assess the quality and relevance of the patient clusters, we applied:

1. **Visualization & Internal Metrics** (e.g., Silhouette Score, Davies-Bouldin Index)  
2. **Survival Outcome Analysis** using **Kaplan-Meier Curves**  
3. **Clinical & Genomic Feature Analysis** across clusters

Additionally, self-supervised contrastive learning methods were applied to refine patient representations by generating augmented views of the multimodal data.

  🧬 Dataset

- **MSK-IMPACT cohort**
- Data types:
  - Clinical features (e.g., age, treatment, gender)
  - Genomic features (binary mutation profiles)

> Note: The dataset is not included in this repository due to privacy concerns. Access it via [MSK-IMPACT project](https://www.mskcc.org/research-areas/programs-centers/impact).

  📊 Tools & Libraries

- `scikit-learn`
- `PyTorch` / `PyTorch Geometric`
- `scipy`, `numpy`, `pandas`
- `seaborn`, `matplotlib`, `lifelines` (for survival analysis)


git clone https://github.com/SepidehNahali/multimodal-cancer-clustering.git
cd multimodal-cancer-clustering
pip install -r requirements.txt
