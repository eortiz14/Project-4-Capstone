# Credit Risk and Machine Learning
## Project Overview

The project is based on credit card default model that seeks to grant a risk score for each customer. The methodology to be implemented will be classification model and a  cross validation technique. Expectations are focused on obtaining insights that help financial banking to build effective credit scores replacing . 
 
Most banks use traditional classification models like logistic model for credit scores because they find them easier to understand than newer machine learning models that have sophisticated algorithms and hypermeters. Trying to change this paradigm i will use a model considered a black box such as the XGBOOST for credit card defaultees, the objetive is get good metrics and show the interpretability of features(atributes) used to modelling. 

## Problem Statement

Try to predict the default probability in people who wants a new financial product is not a new problem. Banks have always looked for a way to correctly identify those clients who, after the approval and disbursement of the credit, fall into credit default. The credit default happens when a client who have an active credit start not paying  to the point that the bank is forced to lose the loan money and reports the client in the credit bureaus. This situation results unfavorable to the banks in many ways because when the clients dont pay they lose money.

Thats the importance to build robusts models that estimate the probability of credit default in people, the global target for the credit risk model is try to identify those clients whose current features have a high probability of not paying in the future. The features are information that the bank have of every client like monthly income, age, gender, for example. This information is used to build credit risk models and nowdays data mining can collect all the information from social networks, demographic, social and financial the models have better inputs to improve their predictive power.

With the database that we obtained from kaggle at first glance we can see 121 possible predictor variables and the target variable which is those clients with payment difficulties had late payment more than X days on at least one of the first Y installments of the loan in our sample. Applying a part of the CRISP-DM methodoly we are going to understand the data, prepare the data, modeling with the prepare data and finally, evaluate the model. We hope to perform well in the machine learning model.

## Metrics

The metrics that we will use are AUC and ROC CURVE. AUC is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve. We expect obtain an AUC over 0.7


## Analysis

For this project we are going to use a credit card defaultees database. The target feature is: client with payment difficulties he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample. Also the database contains 121 atributes(columns) with sociodemographic and credit information data.

At first look for the dataset we see the next table that give us a preliminary description 

![image](https://user-images.githubusercontent.com/88516507/140238277-708d7aa8-d4ce-4f8f-85c8-fff672e0b723.png)

This description table show us 2 things, the firt one is that we have categorical and numerical features and the second one, each type of feature have missing values(Nan) so we must handle with it and fix it. Due to the large number of variables we will treat the numerical and the categories features separately.

Trying to optimize the process of data analysis/data preparation in firt we will drop the features that have mora than 50% of missing values. In this process we drop 41 features, the next table shows some features with more than 50% missing values.

![image](https://user-images.githubusercontent.com/88516507/140239190-97f350bf-0a3e-4f90-b0a3-d8eb6a55b546.png)

Now we can deal with numerical and categorical features more faster. 

#### Categorical Features

The database after the first cleanup has 45 categorical features, ploting each feature we see the next behavior in some features:

![image](https://user-images.githubusercontent.com/88516507/140244992-5fbb281c-ade6-4854-a06f-00d0ef65df8c.png)

As we can see, although some features don't have missing value but the distribution of levels are unfuncional so we drop the categorical features with this levels distribution. 
We also find some attributes that are not related to the purpose of the model such as the variable 'WEEKDAY_APPR_PROCESS_START' that indicates on which day of the week did the client apply for the loan, features unrelated to the purpose of the model were also removed. At the end of this process we still have 14 categorical features.

#### Numerical Features

At beginning of this step in data preparatarion we have 34 numerical features.In firt place we start looking the percentage of nan values in columns and we found this:

![image](https://user-images.githubusercontent.com/88516507/140248058-b3a367f8-31c5-468a-a809-3839d9cde60a.png)

As we can see, some features have a high percentage and others a low percentage. For the high percentage features we drop the column and for the others we will eliminate the rows or impute values. After drop columns, drop rows and impute values we finish with 23 numerical features. At this point we only need to do a correlation analysis of variables to make sure that we do not have 2 or more variables that behave the same, looking the next plot we se high correlation between some features so we drop one of them.

![image](https://user-images.githubusercontent.com/88516507/140249065-92d0ccc2-398c-4e9b-b0dc-ce5740265726.png)

At the end of the data preparation we obtain 34 features and 245891 sku_id(rows).

## Methodology

### Data Preprocessing

Now we have the definitive database but is not ready to put into the model. Build pipelines for the data will help us to get it ready.  

```markdown
# get the categorical feature names
categorical_features = X.select_dtypes("object").columns.to_list()
# get the numerical feature names
numerical_features = X.select_dtypes("int64").columns.to_list()
numerical_features = numerical_features+X.select_dtypes("float64").columns.to_list()

# create the steps for the categorical pipeline
categorical_steps = [
    ('cat_selector', FeatureSelector(categorical_features)),
    ('encoder',One_Hot_Encoder)
]
# create the steps for the numerical pipeline
numerical_steps = [
    ('num_selector', FeatureSelector(numerical_features)),
    ('std_scaler', StandardScaler()),
]
# create the 2 pipelines with the respective steps
categorical_pipeline = Pipeline(categorical_steps)
numerical_pipeline = Pipeline(numerical_steps)
```
The code above let us preprocess numerical features with Standard Scaler and categorical features with One Hot Enconder that inside have 'sparse = False' that help us to drop a one level if the feature have only 2.

Also we must split the database in training base and test base, 70& and 30% respectively.

### Implementation XG-Boost
For the modeling process we are gonna use the Extreme Gradient Boost Classifier (XG-Boost). This model use ensembles that are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models. This is a type of ensemble machine learning model referred to as boosting.

Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm. This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network.

The model have the option to tunning hyperparameter. In this case we will tune this:
             ![image](https://user-images.githubusercontent.com/88516507/140250463-916e9a30-6c56-4a11-a7d5-5b6aff16125f.png)

### Refinement

To improve the algorithm, we use cross-validation for training. We divide into 10 pleats, to help avoid overfitting. We also adjust the hyperparameters according to the table shown above.

```markdown
param_grid = {
    "learning_rate": [0.1, 0.01],
    "colsample_bytree": [0.6, 0.8],
    "subsample": [0.6, 0.8],
    "max_depth": [2, 4, 6],
    "n_estimators": [100, 200, 300],
    "reg_lambda": [1, 1.5, 2],
    "gamma": [0, 0.1, 0.3],
}
```
The algorithm select the hyperparameters that optimize the evaluation metrics "error" and "auc".

## Results

In the following graph we can see the results of the algorithm:

![image](https://user-images.githubusercontent.com/88516507/140252406-0d25effe-f6e6-4bca-a589-15d92fe952ca.png)

We are getting a AUC=0.75, the academic articles say that if an AUC is above 0.7 the model is okay.

As we said at the top of the article, in the financial industry they prefer not to use this type of machine learning model because they think that their interpretability is complex. With the help of the 'plot.importance' function and the SHAP package we will show that this is not the case.

In first place we can use 'plot.importance' for plot the 'Feature Weight', 'Split Mean Gain' and  'Sample Coverage'. The meaning of each one are:

* Feature Weight: The weight of each feature in the model

* Split Mean Gain: Implies the relative contribution of the corresponding characteristic to the model calculated by taking the contribution of each characteristic for each tree in the model. A higher value for this metric compared to another characteristic means that it is more important for generating a prediction.

* Sample Coverage: Means the relative number of observations related to this characteristic.

![image](https://user-images.githubusercontent.com/88516507/140253402-83133e78-1bc1-43b4-b62e-ac04c30775ea.png)


From the above pictures we can observe that first 8 features have the power of the prediction. EXT_SOURCE_2,EXT_SOURCE_3 and amounts of debt balances are on the top which makes a lot of sense following the logic of the financial industry and the approval of new credit products. EXT_SOURCE_2 and EXT_SOURCE_3 are tow differents normalized score from external data source.  Also we see in the middle barplot something interesting, the feature 'NAME_EDUCATION_TYPE' is the highest education the client achieved and the levels 'Secondary / secondary special' and  'Higher education' are helping the trees to split their branches.

Using SHAP package we undestand with more detail the logic of each feature and how its affect the prediction. Lets see the following plot

![image](https://user-images.githubusercontent.com/88516507/140254313-9cd4e865-5d04-48de-8350-def874189faa.png)

As we can see, the graph of the shap values is easy to read. The SHAP values are a transformation of the values that we give to the model and indicate us when the variable is in blue color that this range decreases the prediction of the probability that the client will have difficulties to pay in the future. On the other hand, when the variable is in the pink color, its default probability prediction begins to increase.

## Conclusion

As we can see, these "black box" machine learning models work quite well in the financial industry. We can explain the behavior of all characteristics by analyzing the data before modeling and using plot importance and the shap values after modeling.It is very important when we are going to build a credit risk model that the information of each client is always certified by data mining teams because if we use incorrect information in the training database can have good metrics in the results of the model but when it is implemented, it is possible that we start to see real negative results that will materialize as money losses for the bank.
