import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import jieba
from sklearn import metrics

with open('./all_data.csv', 'r', encoding='utf-8') as f:
    text=f.read().split('\n')
#文本
content = [' '.join(list(jieba.cut(item[:item.rindex(',')]))) for item in text]
#标签
label = [int(item[item.rindex(',')+1:]) for item in text]
#7：3划分训练集测试集
x_train,x_test,y_train,y_test = train_test_split(content,label,test_size=0.3, random_state=33)
# 使用 TfidfVectorizer初始化向量空间模型--创建词袋
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern=r'(?u)\b\w+\b')
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

def svm_train(x_train, x_test, y_train, y_test):
    SVM = SVC()
    #参数选择
    parameters = [{'kernel': ['rbf'], 'gamma': [1], 'C': [44]}]
    grid_search = GridSearchCV(SVM, param_grid=parameters, cv=4, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    #保存模型
    joblib.dump(grid_search, './model.pkl')

    predict_test = grid_search.predict(x_test)
    actual = np.array(y_test)
    m_precision = metrics.accuracy_score(actual, predict_test)
    print("准确率：", m_precision)
    print("The best parameters are %s with a score of %0.2f" % (grid_search.best_params_, grid_search.best_score_))

def predict():
    grid_search = joblib.load('./model.pkl')
    docs = ['太卡了', '包装很美', '屏幕漂亮', '快递很快', '流畅性能好']
    zte = [' '.join(list(jieba.cut(i))) for i in docs]
    tdm3 = vectorizer.transform(zte)
    predict_test = grid_search.predict(tdm3)
    print(predict_test)

#训练模型和保存模型
#准确率： 0.9578121897955132
#训练完可以将svm_train(x_train, x_test, y_train, y_test)注释掉节约预测时间
svm_train(x_train, x_test, y_train, y_test)
#预测方法示例
predict()
