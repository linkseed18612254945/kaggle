import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

TARGET_LABEL = 'SalePrice'
INDEX_LABEL = 'Id'
SEP = '-' * 80

def description():
    # 观察房价的基本分布情况
    sns.distplot(train_data[TARGET_LABEL])
    # plt.show()

    # 观察特征属性的基本信息
    train_data.info()
    print(SEP)
    print(train_data.head(8))

    print(SEP)
    print(train_data.corr())
    too_many_null_feature = ['Alley', 'PoolQC', 'MiscFeature']
    low_corr_numberical_feature = ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', '3SsnPorch', 'MiscVal']
    categorical_feature = ['MSZoning', 'Street', 'LotShape', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood']
    sorted_feature = ['LandContour']
    numberical_feature = ['MSSubClass', 'LotFrontage', 'LotArea']





if __name__ == '__main__':
    train_data = pd.read_csv('./house_price/data/train.csv')
    test_data = pd.read_csv('./house_price/data/test.csv')
    all_data = train_data.append(test_data)

    description()