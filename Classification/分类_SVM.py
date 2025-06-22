import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
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
svr_pipeline=Pipeline([
    ('preprocessor',preprocessor),
    ('svr',SVR(kernel='rbf'))
    ])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
svr_pipeline.fit(x_train,y_train)
y_pred=svr_pipeline.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print(f"MSE OF SVR(UNDER RBF)：{mse:.3f}")

param_grid={
    "svr__C":[0.1,1,10,100],
    'svr__gamma':['scale','auto',0.01,0.1,1]
}

grid_search=GridSearchCV(
    svr_pipeline,param_grid,cv=5,scoring='neg_mean_squared_error'
)
grid_search.fit(x_train,y_train)
print(f"最优参数组合：{grid_search.best_params_}")
print(f"优化后的MSE：{-grid_search.best_score_:.3f}")