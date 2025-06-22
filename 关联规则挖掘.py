#apriori算法应用
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
df=pd.read_csv("synthetic_chemistry_dataset.csv")
print(df.columns)
df['binned concentrartion']=pd.qcut(df['concentration_mol_per_l'],q=3,labels=['low con','medium con','high con'])
df['binned temperature']=pd.cut(df['temperature_C'],bins=[10,30,50,100],labels=['cold','miderate','hot'])
df['binned conductivity']=pd.qcut(df[ 'conductivity_S_per_cm'],q=3,labels=['low conductivity','medium conductivity','high conductivity'])
print(df.head())
transactions=df[['salt', 'solvent','binned concentrartion', 'binned temperature','binned conductivity']].astype(str).values.tolist()
te=TransactionEncoder()
te_ary=te.fit(transactions).transform(transactions)
df_enc=pd.DataFrame(te_ary,columns=te.columns_)
freq_items=apriori(df_enc,min_support=0.05,use_colnames=True)
rules=association_rules(freq_items,metric='confidence',min_threshold=0.05)
rules=rules[
    (rules['consequents']==frozenset({'medium conductivity'}))&#frozenset里面直接写列名而不是传参数,不然就一直返回空值了
    (rules['antecedents'].apply(len)>=2)
]
print(rules[['antecedents','consequents','support','confidence','lift']])
