import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
df=pd.read_csv('synthetic_chemistry_dataset.csv')
print(df.head())
x=df[['salt','solvent', 'concentration_mol_per_l',  'temperature_C']]
y=df['conductivity_S_per_cm']
categorical_features=['salt','solvent']
numerical_features=['concentration_mol_per_l',  'temperature_C']
preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical_features),
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ]
)
knn_pipeline=Pipeline([
    ('preprocessor',preprocessor),
    ('knn',KNeighborsRegressor(n_neighbors=5))
])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knn_pipeline.fit(x_train,y_train)
y_pred=knn_pipeline.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error:{mse:.3f}")#这里的mse是针对训练集
param_grid={'knn__n_neighbors':np.arange(1,20)}
grid_search=GridSearchCV(knn_pipeline,param_grid,cv=5,scoring='neg_mean_squared_error')#这里是针对测试集
grid_search.fit(x_train,y_train)
print(f'最佳K值为：{grid_search.best_params_['knn__n_neighbors']}')
print(f"优化后的均方误差为：{-grid_search.best_score_:.3f}")