import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#print(titanic.head())
#print(titanic.info())

x = titanic[['pclass','age','sex']]
y = titanic['survived']
#print(x.info())
#print('\n\n')
x['age'].fillna(x['age'].mean(), inplace = True)
#print(x.info())

#数据分割
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 33)
#使用sklearn里的特征转换器，抽取特征
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
x_train = vec.fit_transform(x_train.to_dict(orient = 'record'))
print(vec.feature_names_)

#对测试数据中的特征进行转换
x_test = vec.transform(x_test.to_dict(orient = 'record'))

#单一决策树
#从sklearn中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#用默认配置初始化决策树分类树
dtc = DecisionTreeClassifier()
#用分割好的模型进行学习
dtc.fit(x_train,y_train)
#用训练好的模型对测试数据进行预测
y_predict = dtc.predict(x_test)

#使用随机森林分布器进行集成模型的训练以及预测分析
from sklearn.ensemble import RandomForestClassifier
#初始化
rfc = RandomForestClassifier()
#拟合
rfc.fit(x_train, y_train)
#预测
rfc_y_pred = rfc.predict(x_test)

#使用梯度提升决策树进行集成模型的训练以及预测分析
from sklearn.ensemble import  GradientBoostingClassifier
#初始化
gbc = GradientBoostingClassifier()
#拟合
gbc.fit(x_train, y_train)
#预测
gbc_y_pred = gbc.predict(x_test)


#从sklearn中导入classification_report
from sklearn.metrics import classification_report

#输出单一树测试集，已经更加准确的精确率，召回率，F1指标
#输出预测的准确性
print('The accuracy of decision tree is ',dtc.score(x_test,y_test))
#输出更详细的分类性能
print(classification_report(y_predict,y_test,target_names = ['died','survived']))

#输出随机树分类器测试集，已经更加准确的精确率，召回率，F1指标
print('The accuracy of random forest classifier is ',rfc.score(x_test,y_test))
print(classification_report(rfc_y_pred,y_test,target_names = ['died','survived']))

#输出随机树分类器测试集，已经更加准确的精确率，召回率，F1指标
print('The accuracy of gradient tree boosting is ',gbc.score(x_test,y_test))
print(classification_report(gbc_y_pred,y_test,target_names = ['died','survived']))