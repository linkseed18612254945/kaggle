import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

TARGET_LABEL = 'sale_quantity'
INDEX_LABEL = 'class_id'
SEP = '-' * 100


def describe():
    train_data.info()
    print(SEP)
    print(train_data.head(8))
    print(SEP)

    id_features = ['brand_id']
    categorical_features = ['type_id', 'level_id', 'department_id', 'TR', 'gearbox_type', 'if_charging', 'price_level'
                            'driven_type_id', 'fuel_type_id', 'newenergy_type_id', 'emission_standards_id', 'if_MPV_id',
                            'if_luxurious_id', 'cylinder_number', 'rated_passenger']
    numberical_features = ['compartment', 'displacement', 'power', 'engine_torque', 'car_length', 'car_width',
                           'car_height', 'total_quality', 'equipment_quality', 'wheelbase', 'front_track', 'rear_track']

    print(train_data[numberical_features + [TARGET_LABEL]].corr())
    print(SEP)


def feature_engine(train_data):
    # brand_id，共36类无缺失值和异常值，直接转为类别Id
    train_data['brand_id'] = category2index(train_data, 'brand_id')

    # type_id, 共4类无缺失值和异常值，直接转为类别Id
    train_data['type_id'] = train_data['type_id'] - 1

    # level_id, 有少量缺失值，将缺失值作为一个新的类别值填充
    train_data['level_id'].fillna(train_data['level_id'].max() + 1, inplace=True)
    train_data['level_id'] = train_data['level_id'] - 1

    # department_id, 共7类无缺失值和异常值，直接转为类别Id
    train_data['department_id'] = train_data['department_id'] - 1

    # TR，合并为6,45(4;5),789(8;7),01
    map_dict = {'4': '4-5', '5': '4-5', '5;4': '4-5', '6': '6', '7': '>7', '8': '>7', '9': '>7', '0': '<1', '1': '<1'}
    train_data['TR'] = train_data['TR'].map(map_dict)
    train_data['TR'] = category2index(train_data, 'TR')

    # gearbox_type, 有一些混合类别需要处理，将MT;AT和AMT视为AT，将AT;DCT视为DCT
    map_dict = {'MT;AT': 'AT', 'AT;DCT': 'DCT', 'AMT': 'AT'}
    train_data['gearbox_type'] = train_data['gearbox_type'].apply(lambda x: map_dict[x] if x in map_dict else x)
    train_data['gearbox_type'] = category2index(train_data, 'gearbox_type')

    # displacement虽然是数字特征，但是只有19个离散值，可以当做类别特征来处理，将低于1000频次的排量类别按是否大于2归为两类
    train_data['displacement'] = train_data['displacement'].apply(lambda x: 2.4 if x > 2.0 else 1.2 if x < 1.4 or x == 1.9 else x)
    train_data['displacement'] = category2index(train_data, 'displacement')

    # if_charging, 只有两类无缺失值和异常值，直接转为id
    train_data['if_charging'] = category2index(train_data, 'if_charging')

    # price有很多缺失值且可用price_level代替，可以直接去掉
    train_data.drop('price', axis=1, inplace=True)

    # price_level，共9类，将5WL与5-8W合并，将50-75W与35-50W合并
    map_dict = {'5-8W': '10WL', '5WL': '10WL', '8-10W': '10WL', '20-25W': '20-35W',
                '25-35W': '20-35W', '35-50W': '35WH', '50-75W': '35WH'}
    train_data['price_level'] = train_data['price_level'].apply(lambda x: map_dict[x] if x in map_dict else x)
    train_data['price_level'] = category2index(train_data, 'price_level')

    # driven_type_id 可以直接使用或将23类合并
    train_data['driven_type_id'] = train_data['driven_type_id'].map({1: 0, 2: 1, 3: 1})

    # fuel_type_id 基本都是一个类，且有缺失值，可以考虑直接去掉
    train_data.drop('fuel_type_id', axis=1, inplace=True)

    # newenergy_type_id 基本都是一个类下，可以直接去掉或将!=1的项合并
    train_data['newenergy_type_id'] = train_data['newenergy_type_id'].map({1: 0, 2: 1, 3: 1, 4: 1})

    # emission_standards_id, 将352合并
    train_data['emission_standards_id'] = train_data['emission_standards_id'].map({1: 0, 2: 1, 3: 1, 5: 1})

    # if_MPV_id， 直接使用
    train_data['if_MPV_id'] = train_data['if_MPV_id'] - 1

    # if_luxurious_id, 直接使用
    train_data['if_luxurious_id'] = train_data['if_luxurious_id'] - 1

    # power, 含有部分异常值, 和equipment_quality相关系数很高, 可以保留也可以去掉
    train_data.loc[train_data['power'] == '81/70', 'power'] = 81
    train_data['power'] = train_data['power'].astype(float)

    # cylinder_number，主要是4缸，将其他的类合并
    train_data['cylinder_number'] = train_data['cylinder_number'].map({4: 0, 6: 1, 3: 1, 0: 1})

    # engine_torque,与power类似并且与power有0.91的相关系数，可以去掉
    # train_data.loc[train_data['engine_torque'] == '155/140', 'engine_torque'] = 155
    # train_data['engine_torque'] = train_data['engine_torque'].astype(float)
    train_data.drop('engine_torque', axis=1, inplace=True)

    # car_length, car_width, car_height 与目标的相关系数都很低，直接去掉
    train_data.drop(['car_length', 'car_width', 'car_height'], axis=1, inplace=True)

    # total_quality与equipment_quality有很高的相关系数，保留一个特征
    train_data.drop('total_quality', axis=1, inplace=True)

    # rated_passenger和并未最多5人和可大于5人
    map_dict = {'5': 0, '4': 0, '4-5': 0}
    train_data['rated_passenger'] = train_data['rated_passenger'].apply(lambda x: map_dict[x] if x in map_dict else 1)

    # wheelbase, front_track, rear_track和目标相关系数很小直接去掉
    train_data.drop(['wheelbase', 'front_track', 'rear_track'], axis=1, inplace=True)

    # sale_date作为主要的控制特征需要将其拆为年和月两个特征
    train_data['year'] = train_data['sale_date'].apply(lambda x: x // 100).astype(int)
    min_year = train_data['year'].min()  # 2012
    train_data['year'] = train_data['year'] - min_year
    train_data['month'] = train_data['sale_date'].apply(lambda x: int(str(x)[-2:]) - 1)
    train_data.drop('sale_date', axis=1, inplace=True)

    return train_data.drop([INDEX_LABEL, TARGET_LABEL], axis=1), train_data[TARGET_LABEL]


def mean_square_error(y_hat, y):
    return np.sqrt(np.mean((y_hat - y) ** 2))


def category2index(data, feature):
    uniques = {c: i for i, c in enumerate(data[feature].unique())}
    return data[feature].apply(lambda x: uniques[x])


def show_feature(feature):
    print(train_data[feature].unique())
    print('Unique number: ', train_data[feature].nunique())
    print(train_data[feature].value_counts())
    sns.distplot(train_data[feature].dropna())
    plt.show()


def linear_regression(x, y):
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    linreg.fit(x, y)
    mse = mean_square_error(linreg.predict(x), y)
    print('Linear train mse: ', mse)
    return linreg, mse


def knn(x, y):
    from sklearn.neighbors import KNeighborsRegressor
    knnr = KNeighborsRegressor()
    knnr.fit(x, y)
    mse = mean_square_error(knnr.predict(x), y)
    print('KNN train mse: ', mse)
    return knnr, mse


def random_forest(x, y):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(x, y)
    mse = mean_square_error(rf.predict(x), y)
    print('Random Forest train mse: ', mse)
    return rf, mse


if __name__ == '__main__':
    train_data = pd.read_csv('yancheng/data/[new] yancheng_train_20171226.csv', na_values='-')
    test_data = pd.read_csv('yancheng/data/yancheng_testA_20171225.csv')
    # describe()
    train_x, train_y = feature_engine(train_data)

    ss = StandardScaler()
    ss.fit(train_x)
    st_train_x = ss.transform(train_x)
    try_train_x = st_train_x[:]
    try_train_y = train_y[:]

    lr_model, lr_score = linear_regression(try_train_x, try_train_y)
    knn_model, knn_score = knn(try_train_x, try_train_y)
    rf_model, rf_score = random_forest(try_train_x, try_train_y)
    print(SEP)

    model_dict = {
        'Model': ['linear', 'knn', 'random_forest'],
        'MSE': [lr_score, knn_score, rf_score]
    }
    models = pd.DataFrame(model_dict).sort_values(by='MSE', ascending=False)
    print('模型训练集分数：')
    print(models)
    print(SEP)

