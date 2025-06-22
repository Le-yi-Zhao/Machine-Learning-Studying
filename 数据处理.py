import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, normalize
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pyecharts.charts import Parallel
import pyecharts.options as opts
from pandas.plotting import parallel_coordinates

df1 = pd.read_csv('synthetic_chemistry_dataset.csv')
print(df1.head())

# ===== 数据标准化 =====
df2 = df1.copy()
scaler = StandardScaler() #示例方法，需要额外先创建
#标准化缩放:特征大小敏感（回归、支持向量机SVM）
df2[["concentration_mol_per_l", "temperature_C"]] = scaler.fit_transform(df1[["concentration_mol_per_l", "temperature_C"]]) #均值为0，方差为1的处理
print(df2.head())

columns_means = df1[['concentration_mol_per_l', 'conductivity_S_per_cm']].mean() #多列需要两对[[]]
print(columns_means)

# 特征缩放：激活函数范围受限（神经网络）
df3 = df1.copy()
min_max_scaler = MinMaxScaler(feature_range=(0,10))
df3[["concentration_mol_per_l", "temperature_C"]] = min_max_scaler.fit_transform(df3[["concentration_mol_per_l", "temperature_C"]])
print(df3[["concentration_mol_per_l", "temperature_C"]].head())
print(df3[["concentration_mol_per_l", "temperature_C"]].mean())

# 非线性变换
from sklearn.preprocessing import QuantileTransformer
df4 = df1.copy()
df5 = df1.copy()
qt1 = QuantileTransformer(output_distribution='normal', random_state=0) #高斯分布：正则化回归，线性模型，SVM
qt2 = QuantileTransformer(output_distribution='uniform', random_state=0) #均匀分布：决策树、聚类
df4[["concentration_mol_per_l", "temperature_C"]] = qt1.fit_transform(df4[["concentration_mol_per_l", "temperature_C"]])
df5[["concentration_mol_per_l", "temperature_C"]] = qt2.fit_transform(df5[["concentration_mol_per_l", "temperature_C"]])

# l1, l2 正则化 & 映射转换
df6 = df1.copy()
print(pd.unique(df6['salt']))
print(pd.unique(df6['solvent']))
#此处我假设我根据专家经验进行有序数值编码，其实是我乱编的，不知道实际情况中遇到该怎么处理？
df6['salt'] = df6['salt'].replace({'LiClO4':0,'LiBF4':1,'LiTFSI':2, 'LiPF6':3})
def solvent_replacement(x):
    if x == 'DMC': return 0
    if x == 'PC': return 1
    if x == "EC": return 2
    if x == "DEC": return 3

df6['solvent'] = df6['solvent'].map(solvent_replacement)
X_normalized1 = normalize(df6, norm='l1')
X_normalized2 = normalize(df6, norm='l2')
print(X_normalized1)
print(X_normalized2)

# ===== 数据归约（PCA 降维） =====
#PCA算法降维 #其实这里变量之间相关性并不是特别强，但还是拿数据集操作一下
X = df1[["concentration_mol_per_l", "temperature_C", 'conductivity_S_per_cm']]
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['pc1', 'pc2'])
plt.scatter(df_pca['pc1'], df_pca['pc2'], alpha=0.7)
plt.xlabel('main component1')
plt.ylabel('main component2')
plt.title("PCA Dedimension")
plt.grid(True)
plt.show()

# ===== 相似度度量 =====
#相似度计算：只关心方向
df7 = pd.DataFrame(X_normalized2)
cos_sin = df7 @ df7.T
print(np.round(cos_sin,3))

# 属性相关性：协方差与相关系数（方向和幅度都需要关注）
#协方差
cov_matrix = df1[["concentration_mol_per_l", "temperature_C"]].cov()
cov_value = cov_matrix.loc["concentration_mol_per_l", "temperature_C"]
print(f"covariance matrix:\n {cov_matrix}")
print(f"covariance value:{cov_value}")

#相关系数
corr_matrix = df1[["concentration_mol_per_l", "temperature_C"]].corr()
corr_value = corr_matrix.loc["concentration_mol_per_l", "temperature_C"]
print(f"correlation matrix:\n {corr_matrix}")
print(f"correlation value:{corr_value}")

# ===== 数据可视化 =====
#数据参量显示
print(df1.describe())
#箱线图显示
df1.boxplot() #这里各列的单位不一样，做出来一坨，还是先标准化处理一下比较好
plt.show()
df8 = df1.copy()
df8[["concentration_mol_per_l", "temperature_C", 'conductivity_S_per_cm']] = scaler.fit_transform(df8[["concentration_mol_per_l", "temperature_C", 'conductivity_S_per_cm']]) #正则化处理
df8.boxplot()
plt.show()

#散点图显示
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df1['concentration_mol_per_l'], df1["temperature_C"], df1['conductivity_S_per_cm'], alpha=0.3, s=1)
ax.view_init(elev=30, azim=45) #调整了高度和角度之后还是一塌糊涂，真是让人不知道该怎么办了
ax.set_xlabel('concentration_mol_per_l')
ax.set_ylabel('temperature_C')
ax.set_zlabel('conductivity_S_per_cm')
plt.title('concentration-temperature-conductivity mixed figure')
plt.show()

#平行坐标图显示：只接受一组属性值，所以我把solvent组删除了，在实际过程中应该怎么做呢？
df9 = df1.drop(columns=['solvent'])
dimensions = [{"dim": i, "name": col} for i, col in enumerate(df9.columns)]
dimensions[0]['type'] = 'category'
data = df9.values.tolist()
parallel = (
    Parallel()
    .add_schema(dimensions)
    .add("data", data)
    .set_global_opts(title_opts=opts.TitleOpts(title='parallel of salt,concentration,temperature,conductivity'))
)
parallel.render_notebook

#上面的代码需要在notebook中生成，但是实在是太慢了，就重新写了一个pandas的
class_column = df9.columns[0]
feature_columns = df9.columns[1:]
df_plot = df9[feature_columns.tolist() + [class_column]]
plt.figure(figsize=(10,6))
parallel_coordinates(df_plot, class_column=class_column, color=plt.cm.Set2.colors, alpha=0.7)
plt.title(f"Parallel Figure:  Class:{class_column}")
plt.grid(True)
plt.tight_layout()
plt.show()

#异常值检测：基于isolation forest方法，但是里面的参数该如何选择呢？
df10 = df1.copy()
model = IsolationForest(contamination=0.05, random_state=23) #这里的参数是我随便选的
df10['outlier_flag'] = model.fit_predict(df10[["concentration_mol_per_l", "temperature_C", 'conductivity_S_per_cm']])
sns.pairplot(df10,
    vars=["concentration_mol_per_l", "temperature_C", 'conductivity_S_per_cm'],
    hue='outlier_flag',
    palette={1:'blue', -1:'red'},
    diag_kind='hist',
    plot_kws={'alpha':0.6}) #注意括号不要换行了，不然就关不住了
plt.suptitle("outlier detection", y=0.9) #稍微抬起来一点，免得压到图,但是1.02的时候已经飞出界面了
plt.show()