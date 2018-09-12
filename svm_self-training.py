import numpy as np
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import jieba
from sklearn import metrics

def svm_train():
    with open('/home/ubuntu/all_data3.csv', 'r', encoding='utf-8') as f:
        text = f.read().split('\n')
    # 文本
    content = [' '.join(list(jieba.cut(item[:item.rindex(',')]))) for item in text]
    # 标签
    label = [int(item[item.rindex(',') + 1:]) for item in text]
    # 7：3划分训练集测试集
    x_train, x_test, y_train, y_test = train_test_split(content, label, test_size=0.9, random_state=33)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    # y_test=np.array(y_test)

    SVM = SVC(probability=True)
    parameters = [{'kernel': ['rbf'], 'gamma': [1], 'C': [44]}]
    grid_search = GridSearchCV(SVM, param_grid=parameters, cv=4, n_jobs=-1, verbose=1)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern=r'(?u)\b\w+\b')
    vectorizer.fit_transform(content)
    for i in range(11):
        x_train_tdm = vectorizer.transform(x_train)
        x_test_tdm = vectorizer.transform(x_test)
        grid_search.fit(x_train_tdm, y_train)
        predict_proba_test = grid_search.predict_proba(x_test_tdm)
        predict = np.argmax(predict_proba_test, axis=1)
        max_proba=np.array([max(i) for i in predict_proba_test])
        split_group = int(max_proba.size * 0.1)
        max_index = np.argsort(max_proba)[-split_group:]
        x_train=np.append(x_train,x_test[max_index])
        y_train=np.append(y_train,predict[max_index])
        x_test=np.delete(x_test,max_index,0)

    #保存模型
    joblib.dump(grid_search, '/home/ubuntu/model.pkl')
def predict():
    with open('/home/ubuntu/all_data3.csv', 'r', encoding='utf-8') as f:
        text = f.read().split('\n')
    # 文本
    content = [' '.join(list(jieba.cut(item[:item.rindex(',')]))) for item in text]
    # 标签
    label = [int(item[item.rindex(',') + 1:]) for item in text]
    # 7：3划分训练集测试集
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern=r'(?u)\b\w+\b')
    content_tdm=vectorizer.fit_transform(content)
    grid_search = joblib.load('/home/ubuntu/model.pkl')
    predict = grid_search.predict(content_tdm)
    actual = np.array(label)
    m_precision = metrics.accuracy_score(actual, predict)
    print("准确率：", m_precision)
    print("The best parameters are %s with a score of %0.2f" % (grid_search.best_params_, grid_search.best_score_))

    docs = ['太卡了', '包装很美', '屏幕漂亮', '快递很快', '流畅性能好']
    zte = [' '.join(list(jieba.cut(i))) for i in docs]
    tdm3 = vectorizer.transform(zte)
    predict_test = grid_search.predict(tdm3)
    print(predict_test)

#训练模型和保存模型
#准确率： 0.9578121897955132
#训练完可以将svm_train(x_train, x_test, y_train, y_test)注释掉节约预测时间
svm_train()
#预测方法示例
predict()


