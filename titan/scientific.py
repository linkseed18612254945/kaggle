import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

"""
科学的数据挖掘特征工程主要包括下列几个任务：
1. Classifying: 对样本的属性特征进行分类，了解各个特征基本性质
2. Correlating: 去分析数据集中哪些特征对我们目标结果有重大影响，可以从统计学上去了解特征与目标之间的相关系数，同时也
                需要关心各个特征彼此之间的相关性，通过关联、合并、修改某些特征来获得更佳的用于训练的特征。
3. Correcting:  分析数据集去找出有错、不准确或误差值，即检查离群点并尝试对这种杂讯进行修改或排除。
4. Completing:  检查特征数据的缺失值，并采取填充空值或摈弃特征的方法进行处理。
5. Converting:  由于最终输入模型的是数值矩阵，需要将某些特征，特别是字符串类的类别、ID特征转化为数值化特征
                如可以采用数字特征或分列OneHot的方式进行处理。
6. Creating:    根据之前对特征数据的分析和理解，可以利用一个或多个已有特征进行转化合并创建新的特征。
7. Charting:    根据属性特征和统计需求选择合适的方法进行可视化呈现。
"""

SEP = '-' * 80 + '\n'
TARGET_LABEL = 'Survived'
INDEX_LABEL = 'PassengerId'


def describe_data():
    # 观察训练数据的类别标签分布是否均衡
    print('类别统计：')
    print(train_data.groupby(TARGET_LABEL).count()[INDEX_LABEL])
    print(SEP)

    # 对特征进行整体的认识，主要观察几个方面：各特征是属于哪一类变量（定类categorical、数值型numerical），各特征的缺失情况
    print('特征概况：')
    print(train_data.info())
    print(SEP)
    print('数据样例：')
    print(train_data.head(3))
    print(SEP)

    # 对特征进行描述性统计
    print('描述性统计：')
    print(train_data.describe(include='all'))
    print(SEP)

    id_features = ['PassengerId', 'Name', 'Ticket']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Cabin']
    numerical_features = ['Age', 'Sibsp', 'Parch', 'Fare']

    # 计算数值型特征与目标之间的相关系数，观察相关系数的绝对值大小
    print('相关系数：')
    print(train_data.drop(INDEX_LABEL, axis=1).corr())
    print(SEP)

    # 对于定类、定序变量或取值较少的离散数值变量可以直接观察每个类别下的目标取值情况
    print('分类特征分布情况：')
    for cf in categorical_features:
        if train_data[cf].nunique() > 10:
            continue
        print('{}-{}：'.format(cf, TARGET_LABEL))
        print(train_data[[cf, TARGET_LABEL]].groupby(cf, as_index=False).mean(), '\n')
    print(SEP)


def feature_engine():
    global all_data
    # ID类特征可以看做是类别极多的分类特征，无法直接使用，
    # 如果ID特征包含某些可以提取的部分就可以将其转化为较少类别的分类特征，如果不能则通常直接舍去。
    all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    normal_title = ['Mr', 'Miss', 'Mrs']
    all_data['SpecialTitle'] = all_data['Title'].apply(lambda x: 1 if x not in normal_title else 0)
    all_data.drop('Name', axis=1, inplace=True)
    all_data.drop('Ticket', axis=1, inplace=True)

    # 类别特征的处理，类别过多的特征采用类似ID类特征的处理方法
    all_data.drop('Cabin', axis=1, inplace=True)
    # Embarked缺失值只有两个，考虑用众数填充，如果缺失较多可以将缺失单独作为一个特征
    all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
    all_data['Embarked'] = all_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})

    # 数值类变量主要是补全空值，根据需求合并一些特征或将连续性数值变量转化为离散的定序变量
    # 补全数值类特征的方法包括：1.在平均数和标准差之间生成随机数。2.使用其他特征信息估计空值特征。3.结合上述两种方法。
    # 利用其他特征估计空值的一个比较简单的方法是找到多个比较相关的类别特征，取这几个类别变量组合的中位数\平均数标准差。
    # 另外随机数的引入可能导致每次训练的模型不同
    # 对于连续性数值特征常常可采用按取值范围进行离散化
    pclass_type = all_data[all_data['Age'].isnull()]['Pclass'].unique()
    sex_type = all_data[all_data['Age'].isnull()]['Sex'].unique()
    for i in pclass_type:
        for j in sex_type:
            ages = all_data[(all_data['Pclass'] == i) & (all_data['Sex'] == j)]['Age']
            guess_age = ages.median()
            # guess_age = random.uniform(ages.mean() - ages.std(), ages.mean() + ages.std())
            all_data.loc[(all_data['Pclass'] == i) & (all_data['Sex'] == j) & (all_data['Age'].isnull()), 'Age'] = guess_age
    all_data['AgeBand'] = pd.qcut(all_data['Age'], 4)
    band_dict = {c: i for i, c in enumerate(all_data['AgeBand'].unique())}
    all_data['AgeBand'] = all_data['AgeBand'].apply(lambda x: band_dict[x])
    all_data.drop('Age', axis=1, inplace=True)

    # Sibsp, Parch两个数值特征有相似的意义，可以做合并转化处理
    all_data['Families'] = all_data['Parch'] + all_data['SibSp'] + 1
    all_data['Along'] = 0
    all_data.loc[all_data['Families'] == 1, 'Along'] = 1
    all_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)

    # Fare缺失较少, 可以用类似Age的处理方式，也可以直接用总体中位数/均值标准差填充
    all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)
    all_data['FareBand'] = pd.qcut(all_data['Fare'], 5)
    band_dict = {c: i for i, c in enumerate(all_data['FareBand'].unique())}
    all_data['FareBand'] = all_data['FareBand'].apply(lambda x: band_dict[x])
    all_data.drop('Fare', axis=1, inplace=True)

    all_data.drop('Title', axis=1, inplace=True)

    # 将类别属性分列
    all_data = pd.get_dummies(all_data, columns=['SpecialTitle', 'Embarked', 'AgeBand', 'Families', 'Along', 'FareBand'])
    print('特征工程后的特征向量：')
    print(all_data.head(5))
    print(SEP)
    return all_data[:train_data.shape[0]].drop([TARGET_LABEL, INDEX_LABEL], axis=1),\
           all_data[:train_data.shape[0]][TARGET_LABEL], \
           all_data[train_data.shape[0]:].drop(TARGET_LABEL, axis=1)


def logistic():
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(train_x, train_y)
    acc_log = round(logreg.score(train_x, train_y) * 100, 3)
    print('Logistic train acc: ', acc_log)
    return logreg, acc_log


def svm():
    from sklearn import svm
    svc = svm.SVC()
    svc.fit(train_x, train_y)
    acc_svc = round(svc.score(train_x, train_y) * 100, 3)
    print('SVM train acc: ', acc_svc)
    return svc, acc_svc


def knn():
    from sklearn.neighbors import KNeighborsClassifier
    knnc = KNeighborsClassifier(n_neighbors=5)
    knnc.fit(train_x, train_y)
    acc_knn = round(knnc.score(train_x, train_y) * 100, 3)
    print('KNN train acc: ', acc_knn)
    return knnc, acc_knn


def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(train_x, train_y)
    acc_nb = round(nb.score(train_x, train_y) * 100, 3)
    print('Naive Bayes train acc: ', acc_nb)
    return nb, acc_nb


def perceptron():
    from sklearn.linear_model import Perceptron
    perc = Perceptron()
    perc.fit(train_x, train_y)
    acc_perc = round(perc.score(train_x, train_y) * 100, 3)
    print('Perceptron train acc: ', acc_perc)
    return perc, acc_perc


def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier()
    dtree.fit(train_x, train_y)
    acc_dtree = round(dtree.score(train_x, train_y) * 100, 3)
    print('Decision tree train acc: ', acc_dtree)
    return dtree, acc_dtree


def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train_x, train_y)
    acc_rf = round(rf.score(train_x, train_y) * 100, 3)
    print('Random Forest train acc: ', acc_rf)
    return rf, acc_rf


def submission(model_name, use_model):
    test_x[TARGET_LABEL] = use_model.predict(test_x.drop(INDEX_LABEL, axis=1))
    res = test_x[[INDEX_LABEL, TARGET_LABEL]].set_index(INDEX_LABEL)
    res.to_csv('./titan/submission/{}.csv'.format(model_name))


if __name__ == '__main__':
    train_data = pd.read_csv('titan/data/train.csv')
    test_data = pd.read_csv('titan/data/test.csv')
    all_data = train_data.append(test_data)

    describe_data()
    train_x, train_y, test_x = feature_engine()

    lr_model, lr_score = logistic()
    svc_model, svc_score = svm()
    knn_model, knn_score = knn()
    nb_model, nb_score = naive_bayes()
    perceptron_model, perceptron_score = perceptron()
    dtree_model, dtree_score = decision_tree()
    rf_model, rf_score = random_forest()
    print(SEP)

    models = pd.DataFrame({
        'Model': ['logistic', 'svm', 'knn', 'naive_bayes', 'perceptron', 'decision_tree', 'random_forest'],
        'Score': [lr_score, svc_score, knn_score, nb_score, perceptron_score, dtree_score, rf_score]
    }).sort_values(by='Score', ascending=False)
    print('模型训练集准确率：')
    print(models)
    print(SEP)

    submission('random_forest', rf_model)
