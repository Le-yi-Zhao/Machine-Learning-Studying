import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
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
k=5
neigh=NearestNeighbors(n_neighbors=k)
nbrs=neigh.fit(x_processed)
distances,indices=nbrs.kneighbors(x_processed)
k_distances=np.sort(distances[:,k-1])
kneedle=KneeLocator(range(len(k_distances)),k_distances,S=1.0,curve='convex',direction='increasing')
estimated_eps=k_distances[kneedle.knee]#kneedle算法的输出结果不是很好，考虑之后引入贝叶斯优化进一步解决相关问题
print(f'eps based on Kneedle estimation:{estimated_eps:.3f}')
estimated_min_samples=max(2,int(len(x_processed)*0.02))
print(f"min_samples under dynamic estimation:{estimated_min_samples}")
dbscan_model=DBSCAN(eps=estimated_eps,min_samples=estimated_min_samples)
dbscan_labels=dbscan_model.fit_predict(x_processed)
print('DBSCAN output:',np.unique(dbscan_labels,return_counts=True))