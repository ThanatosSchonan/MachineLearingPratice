#导入美国波士顿地区房价 模板包
from sklearn.datasets import load_boston
#读取房价的数据存储在boston之中
boston = load_boston()
#输出数据描述
print(boston.DESCR)

#进行必要的数据分割
#实验数据分割器
from sklearn.model_selection import train_test_split
#导入numpy命名为np
import numpy as np
x = boston.data
#导入数据data
y = boston.target
#导入属性target

#x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 33,test_size = 0.25)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33, test_size=0.25)
#分析回归目标的差异
print("The max target value is ",np.max(boston.target))
print("The min target value is ",np.min(boston.target))
print("The average target value is ",np.mean(boston.target))
#numpy中mean为平均值


#导入数据标准化模块
'''from sklearn.preprocessing import StandardScaler
#分别初始化对特征和目标值的标准化器
ss_x = StandardScaler()
ss_y = StandardScaler()'''

from sklearn.preprocessing import StandardScaler
# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
#x_train = ss_x.fit_transform(x_train)
X_train = ss_X.fit_transform(x_train)
#x_test = ss_x.transform(x_test)
X_test = ss_X.transform(x_test)
#y_train = ss_y.fit_transform(y_train.reshape(1, -1))
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
#y_test = ss_y.transform(y_test.reshape(1, -1))
y_test = ss_y.transform(y_test.reshape(-1, 1))
#使用最简单的线性回归模型LinearRegression和SGDRegressor
#接下来的步骤就都与之前类似了，初始化->拟合->预测
#LinearRegression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
#SGDRegressor
from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(x_train,y_train)
sgdr_y_predict = sgdr.predict(x_test)

#使用三种回归评价机制以及两种调用R-squared评价模块的方法，对本节模型的回归性能作出评价


#LR自带评估模块，R2模块，方差，绝对误差
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
print('The value of default measurement of LinearRegression is',lr.score(x_test,y_test))
print('The value of R2 of LinearRegression is ',r2_score(y_test,lr_y_predict))
print('The mean squared error of LinearRegression is ',
      mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print('The mean absoluate error of LinearRegression is', mean_absolute_error( ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict) ) )
