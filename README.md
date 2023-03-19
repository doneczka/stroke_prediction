# stroke_prediction

## **GOAL** – prediction of which person gets a stroke or not

-	Find personal traits of people who might struggle with stroke
-	Fine-tune the model to get the best results

## **Methods and techniques:**
-	EDA as a starting point for drawing conclusions about feature importance in further modeling
-	Visualization techniques for data distributions
-	Statistical inference to check statistical significance in difference between people who had or did not have a stroke
-	Basic feature engineering 
-	Logistic Regression/SVC/k-NN/Decision Tree/Random Forest/for modeling
-	Pipeline introduction
-	GridSearchCV/RandomizedSearchCV for hyperparameter tuning (depending on type of the model)
-	Undersampling and oversampling techniques for dealing with imbalanced dataset
-	XGBoost and CatBoost
-	Feature importance with permutation_importance and shap library

## **Conclusions:**
**Final model for STROKE prediction - summary.**

Best performing model: **XGBoost**

Paremetrs: 

- subsample = 0.5, 
- n_estimators = 100, 
- max_depth = 3, 
- learning_rate = 0.01, 
- colsample_bytree = 0.7, 
- colsample_bylevel= 0.6, 
- scale_pos_weight = 20

Results: 

- For positive class: Precision 14,13%, Recall 85,48%
- For negative class: Precision 99,17%, Recall 77,02%
- Accuracy: 77,38%.

Most important features:

- age
- avg_glucose_level,
- ever_married: no
- smoking_status: unknown
- hypertension
- heart_disease.

## BUSINESS CONSIDERATIONS.

Stroke dataset is an example of imbalanced dataset, with superiority of negative class (ratio: 22.442). 

My goal was to have possible best results with a positive class - highest possible recall, keeping precision at satisfactory level, and also minimizing missclassification of negative class. 

What does it mean in practice?

From ethical perspective, we want to be sure that we will not classify person as not sick while is sick. It would mean, that not performing medical consultation and care for that person, we could lead to that person sooner death (person would obtain no treatement, etc). So we either want to classify correctly person that is sick to provide best medical care and also avoid situation desribed above.

From business perspective we would also like to minimize the number of patients staying at hospital (higher patients ratio and lower cost) and also we want to minimize the number of medical tests to lower the cost. 

Model that I described as the best one, had the best ratio between errors of type I and type II.

I could not find a model that would lower the amount of false negatives, so I could only tune parameters of models that would keep that amount on the same level while minimizing the amount of false positives. 

For example, for Logistic Regression that ratio was: 9/343 = 0,026 while for the model I described as the best it was: 9/322 = 0,028.

