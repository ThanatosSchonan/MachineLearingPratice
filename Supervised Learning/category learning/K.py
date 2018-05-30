from sklearn.datasets import load_iris
iris = load_iris()
iris.data.shape
#print(iris.DESCR)

#分割数据集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 33)

#使用K近邻分类器对iris进行类别预测
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)

#对预测性能进行评估
print('The accuracy of K-Nearest Neighbor Classifier is ',knc.score(x_test, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names = iris.target_names))