'''
特征重要性
'''

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from DataComments import train_set,test_2018Q2

######################################## 特征列和标签列的划分 ########################################
X_train = train_set.drop(columns=['loan_status'])
y_train = train_set['loan_status']
X_test = test_2018Q2.drop(columns=['loan_status'])
y_test = test_2018Q2['loan_status']

######################################## 计算特征重要性 ########################################
# 计算互信息
mi_scores = mutual_info_classif(X_train, y_train)
# 创建一个包含特征名称和互信息值的数据框
mi_df = pd.DataFrame({'Feature': X_train.columns, 'Mutual_Info': mi_scores})

mi_df = mi_df.sort_values(by='Mutual_Info', ascending=False)

# 打印特征重要性
print(mi_df)
mi_df.to_excel('特征重要性.xlsx', index=False)
