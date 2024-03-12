'''
因子分析
pip install factor_analyzer 
'''

import pandas as pd
import numpy as np
from DataComments import train_set, test_2018Q2
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import seaborn as sns
import matplotlib.pyplot as plt

# 因子降维不需要考虑标签列
train_set_dl = train_set.drop(columns=['loan_status'])

# Bartlett's球状检验
chi_square_value, p_value = calculate_bartlett_sphericity(train_set_dl)
chi_square_value, p_value # (2198903.6921247197, 0.0)

# KMO检验
kmo_all,kmo_model=calculate_kmo(train_set_dl)
kmo_model # 0.6576607311676652

'''
Bartlett's球状检验:
    由于p值接近0，小于通常选择的显著性水平（例如0.05），因此我们可以拒绝球状性假设,可以进行因子分析
KMO检验：
    通过结果可以看到KMO大于0.6，也说明变量之间存在相关性，可以进行分析。
'''

faa = FactorAnalyzer(30,rotation=None)
faa.fit(train_set_dl)
 
# 得到特征值ev、特征向量v
ev,v=faa.get_eigenvalues()
ev,v

# 肘部图，选择合适的因子数量
plt.plot(range(1, train_set_dl.shape[1] + 1), ev)
plt.title("Scree Plot")  
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()  
plt.savefig("因子的肘部图",dpi=400)

# 从因子分析结果中获取特征值
eigenvalues, _ = faa.get_eigenvalues()
# 计算每个特征值的方差解释百分数
variance_explained = eigenvalues / np.sum(eigenvalues)
# 计算累计方差解释百分数
cumulative_variance_explained = np.cumsum(variance_explained)
print(cumulative_variance_explained)

# 选择方式： varimax 方差最大化
# 选择固定因子为 30 个，方差解释0.89682112
faa30 = FactorAnalyzer(30,rotation='varimax')
faa30.fit(train_set_dl)
# 公因子方差
faa30.get_communalities()
# 查看旋转后的特征值
faa30.get_eigenvalues()
# 查看成分矩阵  变量个数*因子个数
loadings=faa30.loadings_
loadings_df = pd.DataFrame(loadings, columns=[f'Factor {i}' for i in range(1, 31)], index=train_set_dl.columns)
loadings_df.to_excel("因子旋转后的成分矩阵.xlsx",index=True)

plt.figure(figsize=(25,15))
sns.heatmap(loadings_df.corr(),annot=True,cmap="coolwarm", linewidth=0.5)
plt.title("Factor Loadings Heatmap")
plt.savefig("因子旋转载荷图.png", dpi=400)

# 查看原始特征对因子的贡献度
print(loadings_df)
# 查看因子贡献率
faa30.get_factor_variance()
new = pd.DataFrame(faa30.transform(train_set_dl))
new.to_excel("因子提取后的特征.xlsx",index=True)