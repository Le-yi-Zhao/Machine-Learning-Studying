import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,classification_report
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
# 多元线性回归模型：连续型问题
df1 = pd.read_csv('synthetic_chemistry_dataset.csv')
model1=LinearRegression()
features=['concentration_mol_per_l' ,'temperature_C']
target='conductivity_S_per_cm'
x=df1[features]
y=df1[target]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)
model1.fit(x_train,y_train)
print('coefficient:',model1.coef_)
print('interception:',model1.intercept_)
y_predict=model1.predict(x_test)
print("预测值的均方误差：",mean_squared_error(y_test,y_predict))
print("预测值的决定系数：",r2_score(y_test,y_predict))
#逻辑回归模型：
def mapping_dict(x):
    if x<0.5:
        return("low")
    if x>1.0:
        return('high')
    else:
        return('medium')
df2=df1.copy()
df2['conductivity evaluation']=df2['conductivity_S_per_cm'].map(mapping_dict)
print(df2.head())
le=LabelEncoder()
df2['conductivity_encoded']=le.fit_transform(df2['conductivity evaluation'])
model2=LogisticRegression(multi_class='ovr')#one versus many strategy
features2=['concentration_mol_per_l' ,'temperature_C']
target2='conductivity_encoded'
x2=df2[features2]
y2=df2[target2]
x_train2,x_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.2,random_state=666)
model2.fit(x_train2,y_train2)
y_predict2=model2.predict(x_test2)
print(classification_report(y_test2,y_predict2,target_names=le.classes_))
#多项式回归
df3=df1.copy()
features3=['concentration_mol_per_l' ,'temperature_C']
target3='conductivity_S_per_cm'
x3=df3[features3].values
y3=df3[target3].values
x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,test_size=0.2,random_state=666)
for i in range(1,10):#在7以上出现了过拟合情况
    degree=i
    model3=make_pipeline(PolynomialFeatures(degree=degree,include_bias=False),LinearRegression())
    model3.fit(x3_train,y3_train)
    y3_pred=model3.predict(x3_test)
    print(f'degree={degree}')
    print("MSE",mean_squared_error)
    print("R:",r2_score(y3_test,y3_pred))
#数据集连续自变量只有两个，用Lasso回归有些幽默了，使用Ridge回归
df4=df1.copy()
features4=['concentration_mol_per_l' ,'temperature_C']
target4='conductivity_S_per_cm'
x4=df4[features4].values
y4=df4[target4].values
x4_train,x4_test,y4_train,y4_test=train_test_split(x4,y4,test_size=0.2,random_state=666)
scaler=StandardScaler()
x4_train=scaler.fit_transform(x4_train)
x4_test=scaler.transform(x4_test)
alphas=np.logspace(-4,4,200)
coefs=[]
mse_vals=[]
for alpha in alphas:
    model4=Ridge(alpha=alpha)
    model4.fit(x4_train,y4_train)
    coefs.append(model4.coef_)
    y4_pred=model4.predict(x4_test)
    mse_vals.append(mean_squared_error(y4_test,y4_pred))
coefs=np.array(coefs)
plt.figure(figsize=(8,6))
for i in range(coefs.shape[1]):
    plt.plot(alphas,coefs[:,i],label=features[i])
plt.xscale('log')
plt.xlabel("alpha(log scale)")
plt.ylabel("Coefficient value")
plt.title('Ridge Coefficients vs alpha')
plt.legend(loc='best')
plt.gca().invert_xaxis()
plt.show()
plt.figure(figsize=(8,4))
plt.plot(alphas,mse_vals,marker='o',markersize=3)
plt.xscale('log')
plt.xlabel("alpha(log scale)")
plt.ylabel("Test MSE")
plt.title('Test MSE vs alpha')
plt.gca().invert_xaxis()
plt.show()