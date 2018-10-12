# Titanic: introduction on data preprocessing, exploratory data analysis, and modeling

##This repo explores Titanic data for predicting survival of on-board passengers with particular focus on data preprocessing (eg. data imputer, feature extraction, etc.) and exploratory data analysis. Besides, we also demonstrate  first-level base estimators and second-level stacking models.

#Introduction
  Then "insinkable" Giantic ship Titanic hit icebery during her mainden voyage in 1912. Due to insufficient life boats, the tragedy led to massive claim of lives. Here we are challenged to build machine learning models to predict whether he/she can survive based on the data. The original data includes passengers' bio info (name, sex, age, SibSp, Parch) and travel info (carbin, embarked port, fare, ticket, pclass).

#Data preprocessing
  First, from the original data, what additional features can be extracted for machine learning modeling? The 'Name' contains suffix, last name, first name, and the middle name. 'Suffix' may tell us the passenger's sex, age, sociental class and even educational level, which we should extract it as a unique feature. 'last name' may tell us the family size, but it can also be obtained by adding 'SibSp' and 'Parch'.
  Second, there are missing values in several columns of the original data including Age, Cabin, Embarked, and Fare. However, here it should be careful that value "0" could be true for Age but should be treated as missing value for Fare. Data imputer in columns with missing value is a pre-requisite for further processing.
  1). Since Fare is generally related to PClass, we can imputer the mean fare value of each class to the missing value of the respective class.
  2). 


#Explorotary data analysis

#Modeling

#Summary

