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

##Data load
data_train= pd.read_csv('train.csv')
data_test= pd.read_csv('test.csv')
data_all= pd.concat([data_train, data_test])
data_all = data_all.drop(['Survived'], axis=1)
m_train = data_train.shape[0]
m_test = data_test.shape[0]

##Impute missing data: including pre-processing raw data for imputing missing values in Columns of Embarked, Age and Cabin.

#Missing Fare: intuitively, fare is largely proportional to Pclass, thus, impute the missing fare with mean values of the corresponding Pclass
data_all.loc[data_all['Fare']==0, 'Fare'] = np.nan
data_all['Fare'] = data_all.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))

#The missing values in Embarked, Age and Cabin: no clear-cut ways to impute, will impute based on RandomForest
#To pre-process the raw data in order to run RandomForest for data imputing.

def Simplify_cabins(data):
    data.Cabin = data.Cabin.fillna('N')
    data.Cabin = data.Cabin.apply(lambda x: x[0])
    return data

def NameSuffix(data):
    data['LastName'] = data['Name'].apply(lambda x: x.split(',')[0])
    data['FirstName'] = data['Name'].apply(lambda x: x.split(',')[1])
    data['NameSuffix'] = data['FirstName'].apply(lambda x: x.split('.')[0])
    return data

def Encode_features(data, features):
    for feature in features:
        data[feature] = data[feature].fillna(0)
        le = preprocessing.LabelEncoder().fit(data[feature])
        data[feature] = le.transform(data[feature])
    return data

def Feature_preprocess(data):
    data = NameSuffix(data)
    data = Simplify_cabins(data)
    data = Encode_features(data, ['Cabin', 'Embarked',  'Sex'])
    return data

data_all = Feature_preprocess(data_all)
data_train = Feature_preprocess(data_train) #for later visualization

#Simplify name suffix: suffix with less than 10 counts simplified based sex, family size, and sociental class.
namesuffix = {' Col': 'Rare', ' Don': 'Rare', ' Mme': 'Mrs', ' Major': 'Rare', ' Lady': 'Mrs', ' Sir': 'Mr',
              ' Mlle': 'Miss', ' the Countess': 'Mrs', ' Jonkheer': 'Rare', ' Capt': 'Rare', ' Mr': 'Mr', ' Mrs': 'Mrs',
             ' Miss': 'Miss', ' Master': 'Master', ' Rev': 'Rare', ' Dr': 'Rare', ' Ms': 'Miss', ' Dona': 'Mrs'}
data_all["NameSuffix"] = data_all["NameSuffix"].map(namesuffix)
data_train['NameSuffix'] = data_train['NameSuffix'].map(namesuffix)

data_all.loc[(data_all['NameSuffix']=='Rare')&(data_all['Sex']==0), 'NameSuffix'] = 'Mrs' #Explain in the 2.3 EDA
data_train.loc[(data_train['NameSuffix']=='Rare')&(data_train['Sex']==0), 'NameSuffix'] = 'Mrs'

#Add features FamilySize by combining 'SibSp' and 'Parch'
data_all['FamilySize'] = data_all['SibSp'] + data_all['Parch'] + 1


#Missing Embarked: imputing by RandomForest
data_embarked = data_all[['Embarked', 'Fare', 'Pclass', 'FamilySize']]

data_embarked_exist = data_embarked.loc[(data_embarked.Embarked!=0)]
data_embarked_null = data_embarked.loc[(data_embarked.Embarked==0)]
x_embarked = data_embarked_exist.iloc[:, 1:]
y_embarked = data_embarked_exist.iloc[:, 0]
rfc_e = RandomForestClassifier(n_estimators=100)
rfc_e.fit(x_embarked, y_embarked)
y_embarked_hat = rfc_e.predict(data_embarked_null.iloc[:, 1:])
data_all.loc[data_all.Embarked==0, 'Embarked'] = y_embarked_hat

#Missing Age: imputing by RandomForest

data_all_imputer = data_all.drop(['Name', 'Ticket', 'SibSp', 'Parch', 'LastName', 'FirstName'], axis=1)
data_all_imputer = pd.get_dummies(data_all_imputer)

data_age = data_all_imputer[['Age', 'Pclass', 'Sex', 'Fare', 'FamilySize', 'NameSuffix_Master', 'NameSuffix_Miss', 'NameSuffix_Mrs', 'NameSuffix_Mr', 'NameSuffix_Rare']]
data_age_exist = data_age.loc[(data_all.Age.notnull())]
data_age_null = data_age.loc[(data_all.Age.isnull())]
x_age = data_age_exist.values[:, 1:]
y_age = data_age_exist.values[:, 0]

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_age, y_age)
y_hat_age = rfr.predict(data_age_null.values[:,1:])
data_all.loc[(data_all.Age.isnull()), 'Age'] = y_hat_age
data_all_imputer.loc[(data_all_imputer.Age.isnull(), 'Age')] = y_hat_age

#Missing Cabin: imputing by RandomForest
data_cabin = data_all_imputer[['Cabin', 'Pclass', 'Sex', 'Fare', 'FamilySize', 'Embarked', 'Age', 'NameSuffix_Master', 'NameSuffix_Miss', 'NameSuffix_Mrs', 'NameSuffix_Mr', 'NameSuffix_Rare']]
data_cabin_exist = data_cabin.loc[(data_cabin.Cabin!=7)]
data_cabin_null = data_cabin.loc[(data_cabin.Cabin==7)]
x_cabin = data_cabin_exist.iloc[:, 1:]
y_cabin = data_cabin_exist.iloc[:, 0]
rfc_c = RandomForestClassifier(n_estimators=100)
rfc_c.fit(x_cabin, y_cabin)
y_cabin_hat = rfc_c.predict(data_cabin_null.iloc[:, 1:])
data_all.loc[data_all.Cabin==7, 'Cabin'] = y_cabin_hat

data_all.loc[data_all['Cabin']==8,'Cabin'] = data_all['Cabin'].mode()[0]

##Features transform for classification algorithm and select features for modeling

def BinFares(data):
    bins = (0,7.9,14.4,31,1000)
    group_names=['Economy', 'Bussiness', 'Luxury', 'Royal']
    categories = pd.cut(data.Fare, bins, labels=group_names)
    data.Fare = categories
    return data

def BinFamilySize(data):
    bins = (0,1,2,4,11)
    group_names=['Alone', 'Couple', 'SmallFamily', 'LargeFamily']
    categories=pd.cut(data.FamilySize, bins, labels=group_names)
    data.FamilySize=categories
    return data

def bin_transform(data):
    data = BinFares(data)
    data = BinFamilySize(data)
    return data

log_transform_features = ['Age']
for feature in log_transform_features:
    data_all[feature] = np.log1p(data_all[feature])

data_all = bin_transform(data_all)

#Select features for ML and Categorical feature transform by OneHotEncoding

data_all = data_all.drop(['Ticket', 'Name', 'SibSp', 'Parch', 'FirstName', 'LastName', 'Cabin'], axis=1)

#Encode categorical features
dummy_features = ['Embarked', 'NameSuffix']

for feature in dummy_features:
    data_all[feature] = data_all[feature].apply(str)

data_all = pd.get_dummies(data_all)

#split training data into training and validation sets
x = data_all.iloc[:m_train,:].drop(['PassengerId'], axis=1)
y = data_train['Survived']

data_test = data_all.iloc[m_train:,:].drop(['PassengerId'], axis=1)

num_test = 0.3

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = num_test, random_state=23)

##Modeling

accuracy_dict = {}
base_pred = []

#Logistic Regression
lr = LogisticRegression(C=1e3, penalty='l2', max_iter=100, fit_intercept=False, solver='liblinear',
                        n_jobs=-1, random_state=0)
lr = lr.fit(x_train, y_train)
y_hat_lr = lr.predict(x_test)
accuracy_dict['LogisticRegression'] = accuracy_score(y_test, y_hat_lr)
y_pred_lr = lr.predict(data_test)
base_pred.append(y_pred_lr)

# print lr

#Decision Tree
dt = DecisionTreeClassifier(criterion='gini', max_depth=3, max_features=None, min_samples_split=3, min_samples_leaf=2,
                            random_state=0)

dt = dt.fit(x_train, y_train)
y_hat_dt = dt.predict(x_test)
accuracy_dict['DecisionTree'] = accuracy_score(y_test, y_hat_dt)
y_pred_dt = dt.predict(data_test)

# print dt
base_pred.append(y_pred_dt)

#Random Forest
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='gini', max_features='sqrt', min_samples_split=2, n_jobs=-1, verbose=1)
# parameters = {'n_estimators':[100,50],
#               'max_features':['auto'],
#               'criterion': ['gini'],
#               'max_depth': [2,3,5],
#               'min_samples_split': [2,3],
#               'min_samples_leaf': [1,2]
#              }
# acc_scorer = make_scorer(accuracy_score)

# rfc = GridSearchCV(rfc, parameters, cv=3, scoring = acc_scorer, n_jobs = -1, verbose = 1)
rfc = rfc.fit(x_train, y_train)
y_hat_rfc = rfc.predict(x_test)
accuracy_dict['RandomForest'] = accuracy_score(y_test, y_hat_rfc)
y_pred_rf = rfc.predict(data_test)

# print rfc
base_pred.append(y_pred_rf)

#Extra Tree
et = ExtraTreesClassifier(n_estimators=100, max_features='sqrt', max_depth=5, criterion='gini', random_state=0, n_jobs=-1,verbose=1)

# params = {'n_estimators': [100,1000,10000], 'max_features': ['sqrt', 'log2'], 'max_depth': [2,3,5],
#           'min_samples_split': [2,3,5], 'min_samples_leaf': [1,2,3]}

# et = GridSearchCV(et, params, cv=3, n_jobs=-1, verbose=1)
et=et.fit(x_train, y_train)
#print etc.feature_importances_
y_hat_etc=et.predict(x_test)
accuracy_dict['ExtraTrees'] = accuracy_score(y_test, y_hat_etc)
y_pred_et = et.predict(data_test)
# print et
base_pred.append(y_pred_et)

#Gradient Boosting
gb = GradientBoostingClassifier(learning_rate=0.02, n_estimators=100, max_depth=3, max_features = 'auto', min_samples_split=3, loss='deviance')

# params = {'loss': ['deviance'], 'learning_rate': [0.02, 0.1, 0.2], 'n_estimators': [100, 1000], 'max_depth': [2,3,5],
#           'min_samples_split': [2,3]}
# gb = GridSearchCV(gb, params, cv=3, n_jobs=-1, verbose=1)
gb = gb.fit(x_train, y_train)
#print gbc.feature_importances_
y_hat_gb = gb.predict(x_test)
accuracy_dict['GradientBoosting'] = accuracy_score(y_test, y_hat_gb)
y_pred_gb = gb.predict(data_test)
# print gb
base_pred.append(y_pred_gb)

#AdaBoost
ada = AdaBoostClassifier(learning_rate=0.2, n_estimators=500, random_state=0)
ada = ada.fit(x_train, y_train)
y_hat_ada = ada.predict(x_test)
y_pred_ada = ada.predict(data_test)

accuracy_dict['AdaBoost'] = accuracy_score(y_test, y_hat_ada)
# print ada
base_pred.append(y_pred_ada)

#SVM
svc = SVC(kernel='rbf', gamma='auto', C=1)

svc = svc.fit(x_train, y_train)
y_hat_svc = svc.predict(x_test)
y_pred_svc = svc.predict(data_test)

accuracy_dict['SVC'] = accuracy_score(y_test, y_hat_svc)
# print svc
base_pred.append(y_pred_svc)

#XGBoost

xgbc = xgb.XGBClassifier(base_score=0.5, max_depth=2, learning_rate=0.6, n_estimators=1000, objective = 'binary:logistic',
                        booster = 'gbtree', reg_alpha=0.2, reg_lambda=1, random_state=0)
# xgb = xgb.XGBClassifier()
# params = {'max_depth': [2,3,6], 'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [100,1000],
#           'objective': ['binary:logistic'], 'booster': ['gbtree'], 'reg_alpha': [0.1, 0.2]}
# xgbc = GridSearchCV(xgbc, params, n_jobs=-1)
xgbc = xgbc.fit(x_train, y_train)
y_hat_xgb = xgbc.predict(x_test)
y_pred_xgb = xgbc.predict(data_test)
accuracy_dict['XGBoost'] = accuracy_score(y_test, y_hat_xgb)
#print xgbc
base_pred.append(y_pred_xgb)

#LightGBM

lgb = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, max_depth=3, min_child_samples=2, n_estimators=100,
             objective='binary', random_state=0)
lgb = lgb.fit(x_train, y_train)
y_hat_lgb = lgb.predict(x_test)
y_pred_lgb = lgb.predict(data_test)
accuracy_dict['LightGBM'] = accuracy_score(y_test, y_hat_lgb)
# print lgb
base_pred.append(y_pred_lgb)

base_pred.append(data_all.iloc[m_train:, 2])
base_pred = pd.DataFrame(data=base_pred).T
base_pred = base_pred.rename(columns = {0:'Logistic', 1:'DecisionTree', 2:'RandomForest', 3:'ExtraTree',
                                        4:'GradientBoost', 5:'AdaBoost', 6:'SVM', 7:'XGBoost', 8:'LightGBM', 9:'PassengerId'})
base_estimators = base_pred.columns.tolist()
base_estimators.remove('PassengerId')

for base in base_estimators:
    sub = pd.DataFrame(data={'PassengerId':base_pred.PassengerId, 'Survived': base_pred[base]})
    sub.to_csv('sub_base_%s.csv' % base, index=False)


#output feature importances
feature_importance = []
models = [dt, rfc, et, gb, ada, xgbc, lgb]
features = x_train.columns.tolist()
for base in models:
    feature_importance.append(base.feature_importances_)
feature_importance.append(features)
feature_importance = pd.DataFrame(data=feature_importance).T
feature_importance = feature_importance.rename(columns={7:'feature', 0:'DecisionTree',1: 'RandomForest', 2:'ExtraTree',
                                                        3:'GradientBoost', 4:'AdaBoost', 5:'XGBoost', 6: 'LightGBM'}).sort_values('DecisionTree', ascending=False)
#feature_importance