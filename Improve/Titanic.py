import pandas as pd

#从互联网读取目标数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

#分离数据特征和预测目标
y = titanic['survived']
x = titanic.drop(['row.names','name','survived'],axis = 1)

#对缺失数据进行填充
x['age'].fillna(x['age'].mean(),inplace = True)
x.fillna('UNKNOWN',inplace = True)

#分割数据
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 33)

#类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient = 'record'))
x_test = vec.transform(x_test.to_dict(orient = 'record'))

#输出处理后特征向量的维度
print(len(vec.feature_names_))