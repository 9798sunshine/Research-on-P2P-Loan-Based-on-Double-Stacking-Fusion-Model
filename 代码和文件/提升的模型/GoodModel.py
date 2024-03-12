# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, cohen_kappa_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# 加载数据
data = pd.read_excel("data.xlsx")
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建基模型
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
gbdt_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
catboost_model = CatBoostClassifier(iterations=50, random_state=42, verbose=0)
lgbm_model = LGBMClassifier(n_estimators=50, random_state=42)

# 构建Stacking模型
estimators = [('ada', ada_model), ('gbdt', gbdt_model), ('catboost', catboost_model), ('lgbm', lgbm_model)]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# 训练基模型
ada_model.fit(X_train, y_train)
gbdt_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)

stacking_model.fit(X_train, y_train)

# 预测
ada_pred = ada_model.predict(X_test)
gbdt_pred = gbdt_model.predict(X_test)
catboost_pred = catboost_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)
stacking_pred = stacking_model.predict(X_test)

# 评估模型
print("AdaBoost - RMSE:", mean_squared_error(y_test, ada_pred, squared=False))
print("AdaBoost - Accuracy:", accuracy_score(y_test, ada_pred))
print("AdaBoost - F1 Score:", f1_score(y_test, ada_pred, average='weighted'))
print("AdaBoost - Kappa Score:", cohen_kappa_score(y_test, ada_pred))
print('\n')
print("GBDT - RMSE:", mean_squared_error(y_test, gbdt_pred, squared=False))
print("GBDT - Accuracy:", accuracy_score(y_test, gbdt_pred))
print("GBDT - F1 Score:", f1_score(y_test, gbdt_pred, average='weighted'))
print("GBDT - Kappa Score:", cohen_kappa_score(y_test, gbdt_pred))
print('\n')
print("CatBoost - RMSE:", mean_squared_error(y_test, catboost_pred, squared=False))
print("CatBoost - Accuracy:", accuracy_score(y_test, catboost_pred))
print("CatBoost - F1 Score:", f1_score(y_test, catboost_pred, average='weighted'))
print("CatBoost - Kappa Score:", cohen_kappa_score(y_test, catboost_pred))
print('\n')
print("LGBM - RMSE:", mean_squared_error(y_test, lgbm_pred, squared=False))
print("LGBM - Accuracy:", accuracy_score(y_test, lgbm_pred))
print("LGBM - F1 Score:", f1_score(y_test, lgbm_pred, average='weighted'))
print("LGBM - Kappa Score:", cohen_kappa_score(y_test, lgbm_pred))
print('\n')
print("Stacking - RMSE:", mean_squared_error(y_test, stacking_pred, squared=False))
print("Stacking - Accuracy:", accuracy_score(y_test, stacking_pred))
print("Stacking - F1 Score:", f1_score(y_test, stacking_pred, average='weighted'))
print("Stacking - Kappa Score:", cohen_kappa_score(y_test, stacking_pred))
print('\n')