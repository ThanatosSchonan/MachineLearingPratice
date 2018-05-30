import pandas as pd
import numpy as np
column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
              'Marginal Adhesion','Singel Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoil',
              'Mitoses','Class']
data=pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    names=column_names)
#将？替换成为标准缺失值
data = data.replace(to_replace='?',value=np.nan)
#丢弃带有缺失值的数据
data = data.dropna(how = 'any')
print(data[:3])
print(data.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,
                                               random_state = 33)
print(y_train.value_counts())
print(y_test.value_counts())

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#初始化
lr = LogisticRegression()
sgdc = SGDClassifier()
#调用LogisticRegression中的fit函数/模块用来训练模型参数
lr.fit(x_train,y_train)
#使用训练好的模型Lr对x_test进行预测，结果存储在lr_y_predict中
lr_y_predict = lr.predict(x_test)
#调用SGDClassifier中的fit函数/模块来训练模型参数
sgdc.fit(x_train,y_train)
#使用训练好的模型sgdc对X_test进行预测，结果存储在变量sgdc_y_predict中
sgdc_y_predict = sgdc.predict(x_test)

from sklearn.metrics import classification_report

#使用评分函数score获得模型在测试集合上的准确性结果
print('Accuracy of LR Classifier:',lr.score(x_test,y_test))
#利用classification_report模块获得LogisticRegression其他三个指标的结果
print(classification_report(y_test,lr_y_predict,target_names = ['Benign','Malignant']))
print("\n")
print('Accuarcy of SGD Classifier:',sgdc.score(x_test,y_test))
print(classification_report(y_test,sgdc_y_predict,target_names = ['Benign','Malignant']))
