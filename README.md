# Titanic: introduction on data preprocessing, exploratory data analysis, feature engineering and modeling

(Titanic_offical.ipynb: step-by-step implementation; Titanic.py: standalone coding.)

#This repo explores Titanic data for predicting survival of on-board passengers with particular focus on data preprocessing (eg. data imputer, feature extraction, etc.), exploratory data analysis and feature engineering.

#Introduction

  Then "insinkable" Giantic ship Titanic hit icebery during her mainden voyage in 1912. Due to insufficient life boats, the tragedy led to massive claim of lives. Here we are challenged to build machine learning models to predict whether he/she can survive based on the data. The original data includes passengers' bio info (name, sex, age, SibSp, Parch) and travel info (carbin, embarked port, fare, ticket, pclass).

#Data preprocessing
  
  First, from the original data, what additional features can be extracted? The 'Name' contains suffix, last name, first name, and the middle name. 'Suffix' may tell us the passenger's sex, age, sociental class and even educational level, which it could be a contributing feature. 'last name' may tell us the family size, but it can also be obtained by adding 'SibSp' and 'Parch'.
  
  Second, check the missing values. There are missing values from columns of Age, Cabin, Embarked, and Fare. However, here it should be careful that value "0" could be true for Age (baby less than 1 year old) but should be treated as missing value for Fare. Data imputer in columns with missing value is a pre-requisite for further processing.
  1). Since Fare is generally related to PClass, we can imputer the mean (or median) fare value of each class to the missing value of the respective class.
  2). There are no clear and easy approach to imputer missing values of Cabin, Embarked and Age. Here we demonstrate imputer of missing values by machine learning models. It should keep in mind that both train and test data should be imputered by the same dataset, namely, the same features, with machine learning models. Please refer to the code on how to implement it.
  3). 

#Explorotary data analysis (EDA)

  EDA is the funniest part of building a model since it enables us to view the insight in a data set, detect outliers / anomalies, extract important variables, test underlying assumptions, etc. Here we will go through them step by step.
   ![titanic_feature_distribution](https://user-images.githubusercontent.com/34787111/46991657-ae81bd00-d0bb-11e8-9729-442ba4c4ef07.png)
  
  Figure 1. Statistical distribution of raw features.
   
  First, we check the statistical distribution of numerical features. As shown in Figure 1, most of the passengers are 20-40 years old with much less at age below 10 and above 50. The right skewed Fare distribution is somewhat consistent with Passengers' distribution over Pclass. Also, most passengers traveled alone and its frequency decays fast as family size increases.
  
  ![titanic_feature_correlation](https://user-images.githubusercontent.com/34787111/46991656-ae81bd00-d0bb-11e8-9ef0-25358df2cf9a.png)
  
  Figure 2. Correlation plot of features.
  
  Second, we are curious whether there are correlation among features. Machine learning models typically perform well with non-strong-correlated features. As shown in Figure 2, as expected, Fare and Pclass shows strong correlation.  Age and Pclass shows some correlation (the older the richer?). This can be confirmed by plotting Age vs. Pclass (or Age vs. Fare), which does show that overall older passengers have lower Pclass and higher Fare. In addition, Age shows correlation with FamilySize. By plotting Age vs FamilySize, it suggests that older passengers tend to travel alone while young passengers more likely travel with one or more family members. The correlation with Survived is the most important since we are predicting survival probability here. We will discuss it in detail later.
  
  ![titanic_categorical_feature_survival](https://user-images.githubusercontent.com/34787111/46991654-ade92680-d0bb-11e8-853e-83e74d0e4bbd.png)
  
  Figure 3. Correlation plots between categorical features and Survived from training dataset.
  
  Third, we start to explore feature correlation with class (survived). As shown in Figure 3 (top left), Female has 4 times higher survival rate than Male, suggesting that Sex is a very strong factor for predicting. Also, Pclass, regardless of Sex, shows strong correlation with survival rate. However, it is surprised that Embarked at #1 has higher survival rate for both male and female, which is against our intuition. In order to uncover this mystery, we plot Embarked vs Pclass (as shown in Figure 4). It clearly shows that there were more Pclass#1 passengers embarked at #1. Coupled with the above observations, we can safely conclude that it is non-uniform Pclass distrition among Embarked contributing to higher survival rate in Embarked#1.
  
![titanic_embarked_pclass](https://user-images.githubusercontent.com/34787111/47059647-22d16480-d17f-11e8-85f2-9a2483ad2119.png)

  Figure 4. Plot of Embarked vs Pclass.
  
  In addition, it seems that NameSuffix also has strong correlation with survival rate. What is insight behind it? Can you decouple it from Sex and reveal new insights? To answer these questions, we made a series of plots between NameSuffix and features (see Figure 5). NameSuffix 'Mr', 'Master' and 'Rare' were only given to male while 'Mrs' and 'Miss' to female. So why 'Master' and 'Rare' have higher survival rate than 'Mr'. 'Master' is in the youngest age group but unfavored in Pclass. However, 'Rare' generally has older age and the lowest Pclass. On the contrary, 'Mr' has the age in adult group, almost the highest Pclass and the lowest Fare, all these are not favored for survival (to be confirmed with Age vs Survival). 
  
  ![titanic_namesuffix](https://user-images.githubusercontent.com/34787111/46991658-ae81bd00-d0bb-11e8-8172-214b48eb4139.png)
  
  Figure 5. Plots of NameSuffix vs Sex, Age, Pclass and Fare.
  
  To date, we still have two numeric features to be explored against Survival. As shown in Figure 6, age lying at baby group (<5) has the highest survival rate; and the survival rate also show linear relationship with binned fare categories.
  
![titanic_age_fare_survival](https://user-images.githubusercontent.com/34787111/47059646-2238ce00-d17f-11e8-847a-618024345d82.png)

  Figure 6. Plots of numeric features vs Survived.
 
 To summarize, features Sex, Pclass / Fare (partly correlates each other), Age, and even NameSuffix (a hybrid feature containing Sex, Pclass, Fare, and Age) will make the most contribution in predicting probablity of survival.
 
#Feature engineering

  Feature engineering bridges raw data with machine learning modeling and is essential for building an intelligent system. After data preprocessing and exploratory data analysis, we could come up with engineering features for modeling. As mentioned, here we created feature 'FamilySize' by combining SipSb and Parch, extracted feature 'NameSuffix' from passengers' name, and extracted feature 'Ticket1' from Ticket. 
  
  As shown in EDA part, the statistical distribution of FamilySize and Fare are highly skewed. Here we introduce binning (quantization) to transforming these numeric features to categorical for eliminating potential adverse effect from extremely large / small values and/or extremely high / low frequencies in the original data. In addition, the distribution of Age is close to bell-shape. Here we statistically transform it to reduce its skewness. To date, we have data ready for machine learning modeling.
  
#Modeling

  We trained the models with base estimators using the processed data. In general, the base models result in accuracy ~82-83% for the train data and ~80% for the test data. We output feature importances from the base models, Sex, Age, Pclass and Fare are all among the top features of these models.
  
#Summary

  We demonstrated a comprehensive solution for the Titanic dataset. Our models predict that Female, Children, and passengers from Pclass_1 (the highest class) and Fare_Royal (the most expensive) have the most likely to survive while adult male has the lowest survival probability. At least partly, this dataset has told us that one century ago the world had already had a societal norm of children and female priorities. Also, it reveals that the nobles had their privileges.
  








