'''
模型的训练
--> 逻辑回归、 神经网络、 支持向量机、 决策树
'''

import pandas as pd
import time
from math import sqrt
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score,mean_squared_error,make_scorer,cohen_kappa_score,log_loss,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from DataComments import train_set, test_2018Q2

######################################## 特征列和标签列的划分 ########################################
X_train = train_set.drop(columns=['loan_status'])
y_train = train_set['loan_status']
X_test = test_2018Q2.drop(columns=['loan_status'])
y_test = test_2018Q2['loan_status']
######################################## 评估指标 ########################################
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))
# 评估指标
scoring = {
    'RMSE': make_scorer(rmse, greater_is_better=False),
    'Kappa': make_scorer(cohen_kappa_score),
    'Accuracy': 'accuracy',
    'F1 Score':'f1_macro'
}
######################################## 支持向量机 ########################################
svm_model = SVC()
# 定义超参数搜索范围
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring=scoring, cv=5, refit='Accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

start_time = time.time() 
best_svm_model = SVC(C=best_params['C'], kernel=best_params['kernel'])
best_svm_model.fit(X_train,y_train)
end_time = time.time()
training_time = end_time - start_time
cross_val_accuracy = cross_val_score(best_svm_model, X_train, y_train, cv=5, scoring='accuracy')
# 输出最佳模型的参数和交叉验证的精度
print(f"最佳的SVM参数组合: {best_params}")  # C: 100  kernel:rbf
print(f"SVM的5折交叉验证精度: {cross_val_accuracy.mean()}") # 0.9050569794586227

y_pred = best_svm_model.predict(X_test)

rmse_value = rmse(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred,average='macro')

# 输出评估指标
print(f"RMSE: {rmse_value}")  # 0.6722783832186492
print(f"Kappa系数: {kappa}") # 0.16157908295632362
print(f"SVM在测试集上的分类准确度: {accuracy_score(y_test, y_pred)}") # 0.8039817232375979
print(f"支持向量机的F1分数: {f1}") # 0.5253055109758681

print(f"SVM最佳参数模型的训练时长: {training_time} seconds") # 93.7378866672516 seconds

######################################## 神经网络 ########################################
nn_model = MLPClassifier(max_iter=5000)  # 修改模型参数根据需要
param_grid = {
    'hidden_layer_sizes': [(8,), (10,), (15, 10), (10, 8)],
    'activation': ['relu', 'logistic', 'tanh'],
}

grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grid, scoring=scoring, cv=5, refit='Accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

start_time = time.time() # 开始计算的时间
best_nn_model = MLPClassifier(max_iter=5000, hidden_layer_sizes=best_params['hidden_layer_sizes'], activation=best_params['activation'])
best_nn_model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time

# 计算交叉验证的精度
cross_val_accuracy = cross_val_score(best_nn_model, X_train, y_train, cv=5, scoring='accuracy')
# 输出最佳模型的参数和交叉验证的精度
print(f"最佳的神经网络参数组合: {best_params}") # hidden_layer_sizes'：(15, 10)，'activation'：'relu'
print(f"神经网络的交叉验证精度: {cross_val_accuracy.mean()}") #  0.9475557766827023

y_pred = best_nn_model.predict(X_test)

rmse_value = rmse(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred,average='macro')

# 输出评估指标
print(f"RMSE: {rmse_value}") # 0.6965037641481809
print(f"Kappa系数: {kappa}") # 0.3876025731792555
print(f"神经网络在测试集上的分类准确度: {accuracy_score(y_test, y_pred)}") # 0.9198825065274151
print(f"神经网络的F1分数: {f1}") # 0.78314109859850916

# 输出训练时长
print(f"神经网络最佳参数模型的训练时长: {training_time} seconds") # 79.1076819896698 seconds
########################################
decision_tree_model = DecisionTreeClassifier()
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(estimator=decision_tree_model, param_grid=param_grid, scoring=scoring, cv=5, refit='Accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
start_time = time.time() # 开始计算的时间
best_decision_tree_model = DecisionTreeClassifier(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
best_decision_tree_model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time

cross_val_accuracy = cross_val_score(best_decision_tree_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"决策树的最佳参数组合: {best_params}") # 'max_depth': 10, 'min_samples_split': 10
print(f"决策树的交叉验证精度: {cross_val_accuracy.mean()}") # 0.9544364532448368

y_pred = best_decision_tree_model.predict(X_test)

rmse_value = rmse(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred,average='macro')

# 输出评估指标
print(f"RMSE: {rmse_value}") # 0.971661117041717
print(f"Kappa系数: {kappa}") # 0.08370505249759064
print(f"决策树在测试集上的分类精度: {accuracy_score(y_test, y_pred)}") # 0.6434073107049608
print(f"决策树的F1分数: {f1}") #  0.4942230122447784
# 输出训练时长
print(f"决策树最佳参数模型的训练时长: {training_time} seconds") # 1.2126028537750244 seconds

######################################## 逻辑回归 ########################################
logistic_model = LogisticRegression(max_iter=5000)
param_grid = {
    'C':[0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'newton-cg']
}

def rmseLoss(y_true, y_pred):
    return sqrt(log_loss(y_true, y_pred))

scoring = {
    'RMSE': make_scorer(rmse, greater_is_better=False),
    'Kappa': make_scorer(cohen_kappa_score),
    'Accuracy': 'accuracy',
    'F1 Score': make_scorer(f1_score)
}

grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, scoring=scoring, cv=5, refit='Accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

start_time = time.time() # 开始计算的时间
best_logistic_model = LogisticRegression(C=best_params['C'], solver=best_params['solver'])
best_logistic_model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

cross_val_accuracy = cross_val_score(best_logistic_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"逻辑回归的最佳参数组合: {best_params}") #'C': 100, 'solver': 'liblinear'
print(f"逻辑回归的交叉验证精度: {cross_val_accuracy.mean()}") # 0.9297760172311993

# 在测试集上进行预测
y_pred = best_logistic_model.predict(X_test)

rmse_value = rmseLoss(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred,average='macro')

# 输出评估指标
print(f"RMSE: {rmse_value}") # 0.9040090574651961
print(f"Kappa系数: {kappa}") # 0.08217045584159366
print(f"逻辑回归在测试集上的分类准确度: {accuracy_score(y_test, y_pred)}") # 0.6585509138381201
print(f"逻辑回归的F1分数: {f1}") #  0.4825279783395378


# 输出训练时长
print(f"逻辑回归最佳模型的训练时长: {training_time} seconds") # 61.1599497795105 seconds