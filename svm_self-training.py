import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib
import jieba
from sklearn import metrics

import time

#long running
#do something other

def svm_train():
    with open('./all_data.csv', 'r', encoding='utf-8') as f:
        text = f.read().split('\n')
    # 文本
    content = [' '.join(list(jieba.cut(item[:item.rindex(',')]))) for item in text]
    content = np.array(content)
    # 标签(包括没标注的和已标注的，标的在左边，未标的设为-1)
    label = [int(item[item.rindex(',') + 1:]) for item in text]
    #label.extend([-1]*len(predict_text))
    label = np.array(label)
    splitline = 10000
    U = np.array([i for i in range(0,len(content))])
    L = U[:splitline]
    U = U[splitline:]
    y_train = label[L]

    #SVM = MultinomialNB()
    SVM = SVC(kernel='rbf', gamma=1, C=44, probability=True)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern=r'(?u)\b\w+\b')
    content = vectorizer.fit_transform(content)
    split_group = 10000
    while True:

        x_train_tdm = content[L]
        x_test_tdm = content[U]
        SVM.fit(x_train_tdm, y_train)

        predict_proba_test = SVM.predict_proba(x_test_tdm)
        predict = np.argmax(predict_proba_test, axis=1)
        max_proba = np.array([max(i) for i in predict_proba_test])
        if U.size < split_group:
            split_group=U.size
            max_index = np.argsort(max_proba)[-split_group:]
            L = np.append(L, max_index)
            y_train = np.append(y_train, predict[max_index])
            x_train_tdm = content[L]
            SVM.fit(x_train_tdm, y_train)
            break

        max_index = np.argsort(max_proba)[-split_group:]
        L = np.append(L,max_index)
        U = np.delete(U,max_index)
        y_train=np.append(y_train,predict[max_index])

    #保存模型
    joblib.dump(SVM, './model.pkl')

def predict():
    with open('./all_data.csv', 'r', encoding='utf-8') as f:
        text = f.read().split('\n')
    # 文本
    content = [' '.join(list(jieba.cut(item[:item.rindex(',')]))) for item in text]
    # 标签
    label = [int(item[item.rindex(',') + 1:]) for item in text]
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern=r'(?u)\b\w+\b')
    content_tdm = vectorizer.fit_transform(content)
    SVM = joblib.load('./model.pkl')
    predict = SVM.predict(content_tdm)
    actual = np.array(label)
    m_precision = metrics.accuracy_score(actual, predict)
    print("准确率：", m_precision)

    docs = ['太卡了', '包装很美', '屏幕漂亮', '快递很快', '流畅性能好']
    zte = [' '.join(list(jieba.cut(i))) for i in docs]
    tdm3 = vectorizer.transform(zte)
    predict_test = SVM.predict(tdm3)
    print(predict_test)

#训练模型和保存模型
#训练完可以将svm_train(x_train, x_test, y_train, y_test)注释掉节约预测时间
svm_train()
#预测方法示例
predict()
