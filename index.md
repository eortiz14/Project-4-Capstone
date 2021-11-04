# Credit Risk and Machine Learning

## Actual context
Try to predict the default probability in people who wants a new financial product is not a new problem. Banks have always looked for a way to correctly identify those clients who, after the approval and disbursement of the credit, fall into credit default. The credit default happens when a client who have an active credit start not paying  to the point that the bank is forced to lose the loan money and reports the client in the credit bureaus. This situation results unfavorable to the banks in many ways because when the clients dont pay they lose money.

Thats the importance to build robusts models that estimate the probability of credit default in people, the global target for the credit risk model is try to identify those clients whose current features have a high probability of not paying in the future. The features are information that the bank have of every client like monthly income, age, gender, for example. This information is used to build credit risk models and nowdays data mining can collect all the information from social networks, demographic, social and financial the models have better inputs to improve their predictive power.

Most banks use traditional classification models like logit for credit scores because they find them easier to understand than newer machine learning models that have sophisticated algorithms and hypermeters. Trying to change this paradigm i will use a model considered a black box such as the XGBOOST for credit card defaultees, the objetive is get good metrics and show the interpretability of features(atributes) used to modelling.

## Getting close to the database

For this project we are going to use a credit card defaultees database. The target feature is: client with payment difficulties he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample. Also the database contains 121 atributes(columns) with sociodemographic and credit information data.

At first look for the dataset we see the next table that give us a preliminary description 
![image](https://user-images.githubusercontent.com/88516507/140238277-708d7aa8-d4ce-4f8f-85c8-fff672e0b723.png)



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
