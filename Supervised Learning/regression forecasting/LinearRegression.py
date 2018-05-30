import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 把数据转化成pandas的形式，在列尾加上房价PRICE
boston_dataset = datasets.load_boston( )
data = pd.DataFrame(boston_dataset.data)
data.columns = boston_dataset.feature_names
data['PRICE'] = boston_dataset.target

# 取出房间数和房价并转化成矩阵形式
print(data.loc[:, 'RM'].values)
print(data.loc[:, 'PRICE'].values)

x = data.loc[:, 'RM'].values
y = data.loc[:, 'PRICE'].values

# 进行矩阵的转置
x = np.array([x]).T
y = np.array([y]).T

# 训练线性模型
l = LinearRegression( )
l.fit(x, y)

# 画图显示
plt.scatter(x, y, s = 10, alpha = 0.5, c = 'green')
plt.plot(x, l.predict(x), c = 'blue', linewidth = '1')
plt.xlabel("X")
plt.ylabel("Y")
plt.show( )