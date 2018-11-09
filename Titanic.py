import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import preprocessing
from scipy.stats import skew
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

##Data loading
def Load_data():
    data_train= pd.read_csv('train.csv')
    data_test= pd.read_csv('test.csv')
    data_all= pd.concat([data_train, data_test])
    data_all = data_all.drop(['Survived'], axis=1)
    m_train = data_train.shape[0]
    return data_train, data_all, m_train

##Data preprocessing
class data_preprocess(object):
    def __init__(self):
        pass

    def FamilySize(self, data):
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        return data

    def Fare_zeros(self, data):
        '''To reconcile the original data containing both nan and 0 for imputer.'''
        data.loc[(data.Fare==0), 'Fare'] = np.nan
        return data

    def Simplify_cabins(self, data):
        data.Cabin = data.Cabin.fillna('N')
        data.Cabin = data.Cabin.apply(lambda x: x[0])
        return data

    def NameSuffix(self, data):
        data['LastName'] = data['Name'].apply(lambda x: x.split(',')[0])
        data['FirstName'] = data['Name'].apply(lambda x: x.split(',')[1])
        data['NameSuffix'] = data['FirstName'].apply(lambda x: x.split('.')[0])
        namesuffix = {' Col': 'Rare', ' Don': 'Rare', ' Mme': 'Mrs', ' Major': 'Rare', ' Lady': 'Mrs', ' Sir': 'Mr',
                      ' Mlle': 'Miss', ' the Countess': 'Mrs', ' Jonkheer': 'Rare', ' Capt': 'Rare', ' Mr': 'Mr',
                      ' Mrs': 'Mrs',
                      ' Miss': 'Miss', ' Master': 'Master', ' Rev': 'Rare', ' Dr': 'Rare', ' Ms': 'Miss',
                      ' Dona': 'Mrs'}
        data['NameSuffix'] = data["NameSuffix"].map(namesuffix)
        data.loc[(data['NameSuffix']=='Rare')&(data['Sex']==0), 'NameSuffix'] = 'Mrs' #only 1 such data and survived
        return data

    def Encode_features(self, data, features):
        for feature in features:
            data[feature] = data[feature].fillna(0)
            le = preprocessing.LabelEncoder().fit(data[feature])
            data[feature] = le.transform(data[feature])
        return data

    def Feature_preprocess(self, data):
        data = self.FamilySize(data)
        data = self.NameSuffix(data)
        data = self.Simplify_cabins(data)
        data = self.Encode_features(data, ['Cabin', 'Embarked',  'Sex'])
        return data

##Data imputering
class data_imputer(object):
    def __init__(self):
        pass

    def Fare_imputer(self, data):
        '''Fare is largely proportional to Pclass, thus, impute the missing fare with mean values of the corresponding Pclass. '''
        data['Fare'] = data.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))
        return data

    def Embarked_imputer(self, data):
        '''imputing by RandomForest.'''
        data_embarked = data[['Embarked', 'Fare', 'Pclass', 'FamilySize']]
        data_embarked_exist = data_embarked.loc[(data_embarked.Embarked!=0)]
        data_embarked_null = data_embarked.loc[(data_embarked.Embarked==0)]
        x_embarked = data_embarked_exist.iloc[:, 1:]
        y_embarked = data_embarked_exist.iloc[:, 0]
        rfc_e = RandomForestClassifier(n_estimators=100)
        rfc_e.fit(x_embarked, y_embarked)
        y_embarked_hat = rfc_e.predict(data_embarked_null.iloc[:, 1:])
        data.loc[data.Embarked==0, 'Embarked'] = y_embarked_hat
        return data

    def Age_imputer(self, data):
        '''Imputing by RandomForest.'''
        data_all_imputer = data.drop(['Name', 'Ticket', 'SibSp', 'Parch', 'LastName', 'FirstName'], axis=1)
        data_all_imputer = pd.get_dummies(data_all_imputer)

        data_age = data_all_imputer[['Age', 'Pclass', 'Sex', 'Fare', 'FamilySize', 'NameSuffix_Master', 'NameSuffix_Miss',
                                     'NameSuffix_Mrs', 'NameSuffix_Mr', 'NameSuffix_Rare']]
        data_age_exist = data_age.loc[(data.Age.notnull())]
        data_age_null = data_age.loc[(data.Age.isnull())]
        x_age = data_age_exist.values[:, 1:]
        y_age = data_age_exist.values[:, 0]

        rfr = RandomForestRegressor(n_estimators=100)
        rfr.fit(x_age, y_age)
        y_hat_age = rfr.predict(data_age_null.values[:,1:])
        data.loc[(data.Age.isnull()), 'Age'] = y_hat_age
        return data

    def Cabin_imputer(self, data):
        '''#Missing Cabin: imputing by RandomForest'''
        data_all_imputer = data.drop(['Name', 'Ticket', 'SibSp', 'Parch', 'LastName', 'FirstName'], axis=1)
        data_all_imputer = pd.get_dummies(data_all_imputer)
        data_cabin = data_all_imputer[['Cabin', 'Pclass', 'Sex', 'Fare', 'FamilySize', 'Embarked', 'Age', 'NameSuffix_Master',
                           'NameSuffix_Miss', 'NameSuffix_Mrs', 'NameSuffix_Mr', 'NameSuffix_Rare']]
        data_cabin_exist = data_cabin.loc[(data_cabin.Cabin!=7)]
        data_cabin_null = data_cabin.loc[(data_cabin.Cabin==7)]
        x_cabin = data_cabin_exist.iloc[:, 1:]
        y_cabin = data_cabin_exist.iloc[:, 0]

        rfc_c = RandomForestClassifier(n_estimators=100)
        rfc_c.fit(x_cabin, y_cabin)
        y_cabin_hat = rfc_c.predict(data_cabin_null.iloc[:, 1:])
        data.loc[data.Cabin==7, 'Cabin'] = y_cabin_hat
        data.loc[data['Cabin']==8,'Cabin'] = data['Cabin'].mode()[0]
        return data

    def Imputer(self, data):
        data = self.Fare_imputer(data)
        data = self.Embarked_imputer(data)
        data = self.Age_imputer(data)
        data = self.Cabin_imputer(data)
        return data

##Features transform for ML
class feature_transform(object):
    def __init__(self):
        pass

    def BinFares(self, data):
        bins = (0,7.9,14.4,31,1000)
        group_names=['Economy', 'Bussiness', 'Luxury', 'Royal']
        categories = pd.cut(data.Fare, bins, labels=group_names)
        data.Fare = categories
        return data

    def BinFamilySize(self, data):
        bins = (0,1,2,4,11)
        group_names=['Alone', 'Couple', 'SmallFamily', 'LargeFamily']
        categories=pd.cut(data.FamilySize, bins, labels=group_names)
        data.FamilySize=categories
        return data

    def Log_transform(self, data):
        log_transform_features = ['Age']    #to reduce data skewness.
        for feature in log_transform_features:
            data[feature] = np.log1p(data[feature])
        return data

    def Data_categorical_encoding(self, data):
        data = data.drop(['Ticket', 'Name', 'SibSp', 'Parch', 'FirstName', 'LastName', 'Cabin'], axis=1)
        categorical_features = ['Embarked', 'NameSuffix']
        for feature in categorical_features:
            data[feature] = data[feature].apply(str)
        data = pd.get_dummies(data)
        return data

    def Transform(self, data):
        data = self.BinFares(data)
        data = self.BinFamilySize(data)
        data = self.Log_transform(data)
        data = self.Data_categorical_encoding(data)
        return data


##Train and test data split
def data_split(data_all, data_train, test_size, random_state):
    '''split training data into training and validation sets'''
    x = data_all.iloc[:m_train,:].drop(['PassengerId'], axis=1)
    y = data_train['Survived']

    data_test = data_all.iloc[m_train:,:].drop(['PassengerId'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_state)

    return x_train,x_test,y_train,y_test, data_test


##Modeling

def model_base(x_train, y_train, x_test, y_test, data_test, data_all):
    accuracy_dict = {}
    base_pred = []
    models = []

    lr = LogisticRegression(C=1e3, penalty='l2', max_iter=100, fit_intercept=False, solver='liblinear', n_jobs=-1,
                            random_state=0)
    dt = DecisionTreeClassifier(criterion='gini', max_depth=3, max_features=None, min_samples_split=3,
                                min_samples_leaf=2,
                                random_state=0)
    rfc = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='gini', max_features='sqrt',
                                 min_samples_split=2,
                                 n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=100, max_features='sqrt', max_depth=5, criterion='gini', random_state=0,
                              n_jobs=-1,
                              verbose=1)
    gb = GradientBoostingClassifier(learning_rate=0.02, n_estimators=100, max_depth=3, max_features='auto',
                                    min_samples_split=3, loss='deviance')
    ada = AdaBoostClassifier(learning_rate=0.2, n_estimators=500, random_state=0)
    svc = SVC(kernel='rbf', gamma='auto', C=1)
    xgbc = xgb.XGBClassifier(base_score=0.5, max_depth=2, learning_rate=0.6, n_estimators=1000,
                             objective='binary:logistic',
                             booster='gbtree', reg_alpha=0.2, reg_lambda=1, random_state=0)
    lgb = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, max_depth=3, min_child_samples=2,
                             n_estimators=100,
                             objective='binary', random_state=0)

    classifiers_base = [lr, dt, rfc, et, gb, ada, svc, xgbc, lgb]

    classifiers_name = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'ExtraTree', 'GradientBoosting',
                        'AdaBoost', 'SVM', 'XGBoost', 'LightGBM']

    for i in range(len(classifiers_base)):
        model = classifiers_base[i]
        model = model.fit(x_train,y_train)
        models.append(model)
        y_hat = model.predict(x_test)
        accuracy_dict[classifiers_name[i]] = accuracy_score(y_test, y_hat)
        y_pred = model.predict(data_test)
        base_pred.append(y_pred)

    base_pred.append(data_all.iloc[m_train:, 1])
    base_pred = pd.DataFrame(data=base_pred).T
    base_pred = base_pred.rename(columns={0: 'Logistic', 1: 'DecisionTree', 2: 'RandomForest', 3: 'ExtraTree',
                                          4: 'GradientBoost', 5: 'AdaBoost', 6: 'SVM', 7: 'XGBoost', 8: 'LightGBM',
                                          9: 'PassengerId'})
    return accuracy_dict, base_pred, models

##What features are important from the models?
def Feature_importances(models):
    models = models[1:6] + models[7:] #drop LogisticRegression and SVM
    feature_importance = []
    features = x_train.columns.tolist()
    for model in models:
        feature_importance.append(model.feature_importances_)
    feature_importance.append(features)
    feature_importance = pd.DataFrame(data=feature_importance).T
    feature_importance = feature_importance.rename(columns={7:'feature', 0:'DecisionTree',1: 'RandomForest', 2:'ExtraTree',
                                                            3:'GradientBoost', 4:'AdaBoost', 5:'XGBoost', 6: 'LightGBM'}).sort_values('DecisionTree', ascending=False)
    return feature_importance


if __name__ == "__main__":
    data_train, data_all, m_train = Load_data()
    data_all = data_preprocess().Feature_preprocess(data_all)
    data_all = data_imputer().Imputer(data_all)
    data_all = feature_transform().Transform(data_all)
    x_train, x_test, y_train, y_test, data_test = data_split(data_all, data_train, test_size=0.3, random_state=0)
    base_accuracy, base_pred, base_models = model_base(x_train, x_test, y_train, y_test, data_test, data_all)
    feature_importance = Feature_importances(base_models)

    print base_accuracy
    print base_pred.shape
    print feature_importance

