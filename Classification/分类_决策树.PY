import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df1 = pd.read_csv('synthetic_chemistry_dataset.csv')
x = df1[['salt', 'solvent', 'concentration_mol_per_l', 'temperature_C']]
y = df1['conductivity_S_per_cm']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['salt', 'solvent'])
    ],
    remainder='passthrough'
)
x_processed = preprocessor.fit_transform(x)
path = DecisionTreeRegressor(random_state=42).cost_complexity_pruning_path(x_processed, y)
ccp_alphas = path.ccp_alphas[:-1]
scores = [cross_val_score(Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42, ccp_alpha=alpha))
]), x, y, cv=5, scoring='neg_mean_squared_error').mean() for alpha in ccp_alphas]

optimal_alpha = ccp_alphas[np.argmax(scores)]
pruned_tree = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42, ccp_alpha=optimal_alpha))
])

pruned_tree.fit(x, y)
predictions = pruned_tree.predict(x)
print("预测结果：", predictions)
plt.figure(figsize=(20, 10))
plot_tree(pruned_tree.named_steps['regressor'],
          feature_names=pruned_tree.named_steps['preprocessor'].get_feature_names_out(),
          filled=True, rounded=True,
          fontsize=10)
plt.show()