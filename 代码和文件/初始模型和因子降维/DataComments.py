'''
在数据预处理之后，我们得到了清洗之后2016年、2017年、2018年在第二季度的数据。
在开始训练模型之前，先划分训练集和测试集：
    训练集： 2016年和2017年第二季度的数据
    测试集： 2018年第二季度的数据

注： 此后的所有模型训练阶段，我们采用的是数据预处理最后一步标准化后得到的数据（保存在ScaledData文件夹下）
'''

import pandas as pd

# 读取训练集
train_2016Q2 = pd.read_excel("./Data/ScaledData/Scaled_Final_process_LoanStats_2016Q2.xlsx")
train_2017Q2 = pd.read_excel("./Data/ScaledData/Scaled_Final_process_LoanStats_2017Q2.xlsx")
# 读取测试集
test_2018Q2 = pd.read_excel("./Data/ScaledData/Scaled_Final_process_LoanStats_2018Q2.xlsx")

# 合并两个训练集
train_set = pd.concat([train_2016Q2, train_2017Q2], axis=0, ignore_index=True)
# 查看合并后的数据信息
print(train_set.info())

# 导出合并后的数据
train_set.to_excel("合并的训练集数据.xlsx", index=False)