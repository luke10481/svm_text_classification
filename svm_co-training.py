import jieba
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from classifiers import CoTrainingClassifier
from sklearn.naive_bayes import MultinomialNB

with open('./all_data.csv', 'r', encoding='utf-8') as f:
    text=f.read().split('\n')
content = [' '.join(list(jieba.cut(item[:item.rindex(',')]))) for item in text]
label = [int(item[item.rindex(',')+1:]) for item in text]
x_train,x_test,y_train,y_test = train_test_split(content,label,test_size=0.9, random_state=33)
x_train.extend(x_test)
del x_test
y_test=len(y_test)*[-1]
y_train.extend(y_test)
del y_test
# 使用 TfidfVectorizer初始化向量空间模型--创建词袋
tfidfvectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern='(?u)\\b\\w+\\b')
x1_train = tfidfvectorizer.fit_transform(x_train)
#
countvectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
x2_train = countvectorizer.fit_transform(x_train)

# 使用 TfidfVectorizer初始化向量空间模型--创建词袋
tfidfvectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern='(?u)\\b\\w+\\b')
content1 = tfidfvectorizer.fit_transform(content)
#
countvectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
content2 = countvectorizer.fit_transform(content)

print('SVM CoTraining')

SVM = SVC(kernel='rbf', gamma=1, C=44, probability=True)
#SVM = MultinomialNB()

svm_co_clf = CoTrainingClassifier(SVM)
svm_co_clf.fit(x1_train, x2_train, y_train)
y_pred = svm_co_clf.predict(content1,content2)
actual = np.array(label)
m_precision = metrics.accuracy_score(actual, y_pred)
print("准确率：", m_precision)

