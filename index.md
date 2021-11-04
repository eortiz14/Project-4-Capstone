# Credit Risk and Machine Learning

## Actual context
Try to predict the default probability in people who wants a new financial product is not a new problem. Banks have always looked for a way to correctly identify those clients who, after the approval and disbursement of the credit, fall into credit default. The credit default happens when a client who have an active credit start not paying  to the point that the bank is forced to lose the loan money and reports the client in the credit bureaus. This situation results unfavorable to the banks in many ways because when the clients dont pay they lose money.

Thats the importance to build robusts models that estimate the probability of credit default in people, the global target for the credit risk model is try to identify those clients whose current features have a high probability of not paying in the future. The features are information that the bank have of every client like monthly income, age, gender, for example. This information is used to build credit risk models and nowdays data mining can collect all the information from social networks, demographic, social and financial the models have better inputs to improve their predictive power.

Most banks use traditional classification models like logit for credit scores because they find them easier to understand than newer machine learning models that have sophisticated algorithms and hypermeters. Trying to change this paradigm i will use a model considered a black box such as the XGBOOST for credit card defaultees, the objetive is get good metrics and show the interpretability of features(atributes) used to modelling. 

## Getting close to the database

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




```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
