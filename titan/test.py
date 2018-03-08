import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

TARGET = 'Survived'


def load_describe():
    """
        首先对读入数据，观察各个特征的特点和数据情况，主要统计一下各个特征与目标的相关性，有一个直观的认知后才能进行接下来的特征工程。
    """
    # 读入训练数据，进行描述性统计，观察特征
    train_data = pd.read_csv('data/train.csv')
    cols = train_data.columns.drop(['PassengerId', TARGET])

    # 属性信息
    print(train_data.info())
    # 简单统计信息
    print(train_data.describe())

    # 目标与各属性的相关系数
    print(train_data.drop('PassengerId', axis=1).corr().loc[TARGET, :])
    # 目标分别与各属性的相关情况
    for c in cols:
        ab_mean = train_data.groupby(c).mean()[TARGET]
        if len(set(train_data[c].values)) <= 10:
            ab_mean.plot.bar()
        else:
            ab_mean.plot()
    return train_data


def feature_engine():
    """
        接下里进行特征工程，包括：特征的取舍，数据补全，提取新特征来代替旧特征，将ID类特征进行分列等
    """
    # 读入测试数据一起进行特征工程
    test_data = pd.read_csv('data/test.csv')
    test_data[TARGET] = 0
    train_test_data = train_data.append(test_data)
    train_test_data.info()

    # 年龄Age，较多缺失，考虑用平均值补全缺失值同时构建一个是否存在缺失的新属性
    mean_age = train_test_data['Age'].mean()
    train_test_data.loc[train_test_data['Age'].isnull(), 'Age_null'] = 1
    train_test_data.loc[train_test_data['Age'].notnull(), 'Age_null'] = 0
    train_test_data['Age'].fillna(mean_age, inplace=True)

    # Cabin，大部分缺失，将其转化为根据是否存在以及首字母类别构建的新属性，然后分列。
    train_test_data['Cabin'] = train_test_data['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else x)
    train_test_data = pd.get_dummies(train_test_data, columns=['Cabin'])

    # Embarked, 只有少量缺失，可以不填充或者用众数来填充
    embarked_mode = train_test_data['Embarked'].mode()
    train_test_data['Embarked'].fillna(embarked_mode, inplace=True)
    train_test_data = pd.get_dummies(train_test_data, columns=['Embarked'])

    # Fare, 只有一个缺失值，取值范围在0~512.32之间，考虑用Pclass和Embark对应的均值补全后进行归一化
    print(train_test_data.groupby(['Pclass', 'Embarked']).mean()['Fare'])
    train_test_data.fillna(14.43, inplace=True)

    # Name，是文本类特征不能直接使用，需要提取具有一定意义的文本特征，这里采用提取出名字中的Mr, Mrs, miss等称谓作为特征
    train_test_data['Name'] = train_test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    train_test_data = pd.get_dummies(train_test_data, columns=['Name'])

    # Parch,SibSp 均无缺失，且是性质接近的数值特征，可以相加为一个特征或不做处理
    train_test_data['Known_people'] = train_test_data['Parch'] + train_test_data['SibSp']

    # PassengerId id类属性，可以直接去掉
    train_test_data.drop('PassengerId', axis=1, inplace=True)

    # Pclass 无缺失，可以直接分列后保留
    train_test_data = pd.get_dummies(train_test_data, columns=['Pclass'])

    # Sex 无缺失, 不做处理直接保留

    # Ticket, 特征较复杂，可以先去除不使用，也可以进行简单的抽取
    train_test_data.drop('Ticket', axis=1, inplace=True)

    train_X = train_test_data[:len(train_data)].drop(TARGET, axis=1)
    train_Y = train_test_data[:len(train_data)][TARGET]
    test_X = train_test_data[len(train_data):].drop(TARGET, axis=1)
    return train_X, train_Y, test_X


def data2vector(data):
    """
    将特征数据转化为标准的数字向量，好输入模型中进行训练，需要进行归一化等处理
    """

if __name__ == '__main__':
    train_data = load_describe()
    train_x, train_y, test_x = feature_engine()