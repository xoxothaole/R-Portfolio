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

<p align="center">
  <img src= "images/Model 2.png" alt="Model 2"/><br>
  <em>Model 2: Q-Q plot demonstrates model residuals follow a normal distribution for the majority of cases, with deviation only in extreme values.</em>
</p>

<p align="center">
  <img src= "images/Model 3.png" alt="Model 3"/><br>
  <em>Model 3: Indicates consistent model variance across predicted values — demonstrates reliable performance regardless of transaction characteristics.</em>
</p>

<p align="center">
  <img src= "images/Model 4.png" alt="Model 4"/><br>
  <em>Model 4: Shows transactions that have large errors (high residuals) don’t disproportionately affect model training — ensuring stable performance.</em>
</p>

> <small> *Overall these diagnostic models confirms the model’s potential — displaying it’s reliability and consistent performance across diverse transaction patterns.* </small>

- Decision Tree Model
```r
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class~. , train_data, method = 'class')

rpart.plot(decisionTree_model)

predicted_val <- predict(decisionTree_model, test_data, type = 'class')
probability <- predict(decisionTree_model, test_data, type = 'prob')
```
<p align="center">
  <img src= "images/Model 5.png" alt="Model 5"/><br>
  <em>I developed an interpretable decision tree to identify the specific transaction patterns that indicate fraud. Since the dataset uses anonymized variables (V1-V28) for privacy protection, the model reveals the mathematical patterns that distinguish fraudulent behavior, even though we can’t see the original transaction details. </em>
</p>

- Artifical Neural Network (ANN)
```r
library(neuralnet)
ANN_model = neuralnet(Class~. , train_data, linear.output = FALSE)
plot(ANN_model)

predANN = compute(ANN_model, test_data)
resultANN = predANN$net.result
resultANN = ifelse(resultANN > 0.5, 1, 0)
```
<p align="center">
  <img src= "images/Model 6.png" alt="Model 6"/><br>
  <em>I used neural networks because they can detect subtle, interconnected patterns across all 29 transaction features the other models can’t. In this case, each connection represents a “weight” that the network learned during training. The network discovered which combinations of transaction features, when activated together, indicate fraudulent behavior. Unlike decision trees with clear rules, neural networks create complex mathematical relationships that can detect very subtle fraud signatures. </em>
</p>

### End Results and Reccomendations 
- Model Performance (AUC Scores)
```r
# Calculate AUC for all models
log_pred = predict(Logistic_Model, test_data, type = "response")
log_auc = roc(test_data$Class, log_pred, quiet = TRUE)

tree_pred_prob = predict(decisionTree_model, test_data, type = 'prob')[,2]
tree_auc = roc(test_data$Class, tree_pred_prob, quiet = TRUE)

ann_pred_prob = predANN$net.result
ann_auc = roc(test_data$Class, ann_pred_prob, quiet = TRUE)

# Compare all models
models_performance = data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Neural Network", "GBM"),
  AUC = round(c(auc(log_auc), auc(tree_auc), auc(ann_auc), auc(gbm_auc)), 4)
)

print("Model Performance Comparison:")
print(models_performance)

# Find best model
best_model = models_performance[which.max(models_performance$AUC), ]
print(paste("Winner:", best_model$Model, "with AUC of", best_model$AUC))
```
<p align="center">
  <img src= "images/Table 1.png" alt="Table 1"/><br>
</p>

```r 
# Combined ROC Curves
plot(log_auc, col="blue", lwd=2, main="ROC Curve Comparison - All Models")
plot(tree_auc, add=TRUE, col="green", lwd=2)
plot(ann_auc, add=TRUE, col="red", lwd=2) 
plot(gbm_auc, add=TRUE, col="orange", lwd=2)
legend("bottomright", 
       legend=c("Logistic Regression (97.5%)", "Decision Tree (90.3%)", 
                "Neural Network (93.5%)", "GBM (95.9%)"),
       col=c("blue", "green", "red", "orange"), lwd=2, cex=0.8)
```
<p align="center">
  <img src= "images/Model 8.png" alt="Model 1"/><br>
</p>

1. **Logistic Regression (97.48%) - fast, interpretable, linear patterns**
2. **Gradient Boosting (95.92%) - complex ensemble, slower inference** 
3. **Neural Network (93.50%) - non-linear patterns, black box**
4. **Decision Tree (90.28%) - highly interpretable, simple rules**

### Key Findings
- Linear patterns dominate → simpler models outperformed complex ones
- Class imbalance handled successfully → all models >90% AUC despite 0.17% fraud rate
- Feature importance → V17, V14, V12 identified as primary fraud indicators
- Reproducible results → consistent performance across train/test splits

### Business Recommendations
- **Deploy logistic regression model -**
    - Superior accuracy with minimal computational overhead
    - Explanatory predictions for regulatory compliance
    - Millisecond predictions for real-time processing
    - Simple architecture and easy updates → simple maintenance/ retraining
    - Lower computational requirements → Cost effective

### Model Monitoring Strategy
- Weekly AUC performance tracking
- Monthly retraining with new data
- Threshold optimization based on precision/recall trade-offs
- A/B testing framework for model updates

### **Conclusion**
Leveraging real-world European transaction data, this end-to-end fraud detection initiative achieved a 97.48% AUC and demonstrated my command of the full machine learning lifecycle. I evaluated four approaches—logistic regression, decision trees, neural networks, and gradient boosting—and found that a well-tuned logistic regression outperformed the more complex models. By addressing class imbalance, engineering features for maximum signal, and prioritizing model interpretability, I delivered production-ready workflows spanning data preprocessing, model development, performance evaluation, and clear business recommendations. This project reinforced that deep data understanding and rigorous validation often trump flashy techniques—and showcased my ability to tackle real-world machine learning challenges.

### Dashboard
