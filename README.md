# Credit Card Fraud

### Objective
To develop and compare multiple machine learning models that can automatically detect fraudulent credit card transactions in real-time, helping financial institutions prevent fraud while minimizing false alarms for legitimate customers. 

### Context
Credit card fraud causes billions in annual losses globally with rates increasing 225% during COVID-19. Rule-based systems can’t keep up with the pace of fraudster’s evolving tactics — thus, the implement of machine learning can continuously retrain on fresh data and spot subtle anomalies across billions of transactions to deliver dynamic risk scores that catch emerging fraud faster, lower chargeback costs, and preserve customer trust. The dataset used in this project is from European credit card transactions from September 2013 containing 284,807 transactions with only 492 fraudulent cases. 

### Tools and Methodologies 
I employed the CRISP-DM framework and carried out all data manipulation, machine learning, evaluation, and visualization in RStudio using R packages. The dataset was split into 80/20 into training and test sets with stratified sampling to preserve the fraud ratio, and k-fold cross-validation was applied on the training portion for robust, unbiased model assessment.


### Data Set
- European cardholders credit card transactions in September 2013 (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data?select=creditcard.csv).

## The Approach and Process 
- Data Manipulation
```r
head(creditcard_data)

creditcard_data$Amount=scale(creditcard_data$Amount) #Standardization
NewData = creditcard_data[, -c(1)]  # Feature selection
head(NewData)
```
> <small> *This process was important to ensure fair algorithm performance and remove irrelevant data. Applied z-score normalization to transaction amounts so prices would be evaluated equally — based on patterns rather than magnitude. Time variable also removed since it had no correlation with fraud patterns.* </small>

- Data Modeling
```r
# Train/Test Split

library(caTools)
set.seed(123) # Reproducibility 
data_sample = sample.split(NewData$Class, SplitRatio = 0.80) # Stratified sampling
train_data = subset(NewData, data_sample == TRUE) # 80% for learning
test_data = subset(NewData, data_sample == FALSE) # 20% for validation
dim(train_data)
dim(test_data)
```
>> <small>* *The 80/20 methodology ensures the 97.48% AUC score — representing genuine predictive capability on unseen data. This provides confidence for production deployment rather than overfitted training performance. I used stratified sampling to maintain the 0.17% fraud ratio across both data sets and set seeds makes the results reproducible for comparison and validation.* </small>

- Logisitc Regression Model
```r
Logistic_Model = glm(Class~., train_data, family = binomial())
summary(Logistic_Model)

plot(Logistic_Model)
```
<p align="center">
  <img src= "images/Model 1.png" alt="Model 1"/><br>
  <em>Model 1: Shows good model performance with random scatter around zero — indicating unbiased predictions.</em>
</p>


### Dashboard
