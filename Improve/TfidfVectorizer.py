'''
使用TfidVectorizer并且去掉停用词的情况下，
对文本特征进行量化的朴素贝叶斯分类性能测试
'''

#对文本特征进行量化的朴素贝叶斯分类性能测试
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = 'all')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size = 0.25, random_state = 33)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(analyzer = 'word',stop_words = 'english')
#去掉停用词
x_tfidf_train = tfidf_vec.fit_transform(x_train)
x_tfidf_test = tfidf_vec.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(x_tfidf_train,y_train)

from sklearn.metrics import classification_report
print(mnb_tfidf.score(x_tfidf_test,y_test))
y_tfidf_predict = mnb_tfidf.predict(x_tfidf_test)
print(classification_report(y_test, y_tfidf_predict,target_names = news.target_names))
