import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
df=pd.read_csv("synthetic_chemistry_dataset.csv")
print(df.columns)
x=df[['salt', 'solvent', 'concentration_mol_per_l', 'temperature_C']]
categorical_features=['salt', 'solvent']
numerical_features=['concentration_mol_per_l', 'temperature_C']
preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical_features),
        ('cat',OneHotEncoder(),categorical_features)
    ]
)
x_processed=preprocessor.fit_transform(x)
def simple_isodata(x,initial_k,max_iter=10,merge_threshold=0.5,split_threshold=1.5):
    k=initial_k
    for i in range(max_iter):
        kmeans=KMeans(n_clusters=k,random_state=42).fit(x)
        centers=kmeans.cluster_centers_
        labels=kmeans.labels_
        cluster_std=[np.std(x[labels==idx]) for idx in range(k)]
        for idx,std in enumerate(cluster_std):
             if std>split_threshold:
                k+=1
        merged=False
        for m in range(k):
            for n in range(m+1,k):
                if np.linalg.norm(centers[m]-centers[n])<merge_threshold:
                    k-=1
                    merged=True
                    break
            if merged:
                break
            if not merged and all(std<=split_threshold for std in cluster_std):
                break
    return labels
isodata_labels=simple_isodata(x_processed,initial_k=3)
print("Output of Isodata clustering:",np.unique(isodata_labels,return_counts=True))