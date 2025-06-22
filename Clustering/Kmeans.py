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
kmeans_model =KMeans(n_clusters=3,init='k-means++',random_state=42)
kmeans_labels=kmeans_model.fit_predict(x_processed)
print("Kmeans聚类结果：",np.unique(kmeans_labels,return_counts=True))
pca=PCA(n_components=2)
x_reduced=pca.fit_transform(x_processed)
plt.figure(figsize=(8,6))
plt.scatter(x_reduced[:,0],x_reduced[:,1],c=kmeans_labels,cmap='viridis',alpha=0.7)
plt.scatter(pca.transform(kmeans_model.cluster_centers_)[:,0],
            pca.transform(kmeans_model.cluster_centers_)[:,1],
            c='red',marker='x',s=200,label='Centroids')
plt.title('Visualization of Kmeans++ clustering')
plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.legend()
plt.grid(True)
plt.show()