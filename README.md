# svm_text_classification
svm.py为普通svm文本分类

svm_self-training.py用到自训练算法(其中一个问题就是在分类中一旦出现错误分类，就会越陷越深，导致精度下降)

svm_co-training.py用到协同训练算法(借用了[jjrob13/sklearn_cotraining的分类包](https://github.com/jjrob13/sklearn_cotraining))

嫌速度慢的可以用sklearn.naive_bayes的MultinomialNB分类器
