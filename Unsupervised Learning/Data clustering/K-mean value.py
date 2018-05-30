import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#导入数据
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
                           header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
                          header=None)

#从训练和测试数据集上都分离出64维度的像素特征与1维度的数字目标
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]
x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

#从sklearn.cluster中导入KMeans模型
from sklearn.cluster import KMeans
#初始化模型，设置聚类中心为10
kmeans = KMeans(n_clusters = 10)
kmeans.fit(x_train)
#逐条判断每个测试图像的中心
y_pred = kmeans.predict(x_test)

#从sklearn导入度量函数库metrics
from sklearn import metrics
#使用ARI进行KMeans聚类性能评估
print(metrics.adjusted_rand_score(y_test, y_pred))

#导入numpy
import numpy as np
#从sklearn.cluster中导入KMeans算法包
from sklearn.cluster import KMeans
#导入silhouette_score用于计算轮廓系数
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#分割出3*2个子图，并且在1号作图
plt.subplot(3,2,1)

#初始化原始数据点
x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
X = np.array(list(zip(x1,x2))).reshape(len(x1),2)

#在1号子图做出原始数据点阵分布
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instance')
plt.scatter(x1,x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3,2,subplot_counter)
    kmeans_model = KMeans(n_clusters = t).fit(X)

    for i,l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color = colors[l], marker = markers[l], ls = 'None')

    plt.xlim([0,10])
    plt.ylim([0,10])
    sc_score = silhouette_score(X, kmeans_model.labels_, metric = 'euclidean')
    sc_scores.append(sc_score)
    plt.title('K=%s,silhouette coefficient = %0.03f'%(t,sc_score))

plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')

plt.show()