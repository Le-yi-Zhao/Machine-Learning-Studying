#因为数据有限，在这里采用2折交叉检验，在原数据集上直接检验bpnn性能
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
df=pd.read_csv('synthetic_chemistry_dataset.csv')
categorical_features=['salt','solvent']
numerical_features=['concentration_mol_per_l',  'temperature_C']
encoder=OneHotEncoder(sparse_output=False)
cat_encoded=encoder.fit_transform(df[categorical_features])
scaler=StandardScaler()
cont_scaled=scaler.fit_transform(df[numerical_features])
x=np.hstack([cat_encoded,cont_scaled])
y=df['conductivity_S_per_cm'].values.reshape(-1,1)
kf=KFold(n_splits=2,shuffle=True,random_state=42)
fold=1
for train_index,test_index in kf.split(x):
    print(f'Fold{fold}')
    x_train_tensor=torch.tensor(x[train_index],dtype=torch.float32)
    y_train_tensor=torch.tensor(y[train_index],dtype=torch.float32)
    x_test_tensor=torch.tensor(x[test_index],dtype=torch.float32)
    y_test_tensor=torch.tensor(y[test_index],dtype=torch.float32)
    class BPNetwork(nn.Module):
        def __init__(self,input_dim):
            super(BPNetwork,self).__init__()
            self.fc1=nn.Linear(input_dim,16)
            self.fc2=nn.Linear(16,8)
            self.fc3=nn.Linear(8,1)
            self.relu=nn.ReLU()
        def forward(self,x_):
            x_=self.relu(self.fc1(x_))
            x_=self.relu(self.fc2(x_))
            x_=self.fc3(x_)
            return x_
    input_dim=x_train_tensor.shape[1]
    model=BPNetwork(input_dim)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.01)
    epochs=50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions=model(x_train_tensor)
        loss=criterion(predictions,y_train_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_predictions=model(x_test_tensor)
        test_loss=criterion(test_predictions,y_test_tensor)
        print(f"Test Loss (Fold{fold}):{test_loss.item():.4f}")
    fold+=1