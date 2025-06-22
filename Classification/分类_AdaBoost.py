import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
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
adaboost_pipeline=Pipeline([
    ('preprocessor',preprocessor),
    ('ada',AdaBoostRegressor(n_estimators=100,random_state=42))
])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
adaboost_pipeline.fit(x_train,y_train)
y_pred_ada=adaboost_pipeline.predict(x_test)
mse_ada=mean_squared_error(y_test,y_pred_ada)
print(f" MSE of AdaBoost:{mse_ada:.3f}")