#对文本特征进行量化的朴素贝叶斯分类性能测试
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = 'all')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size = 0.25, random_state = 33)

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer(analyzer = 'word',stop_words = 'english')
#去掉停用词
x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train,y_train)

print(mnb_count.score(x_count_test,y_test))
y_count_predict = mnb_count.predict(x_count_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_count_predict,target_names = news.target_names))