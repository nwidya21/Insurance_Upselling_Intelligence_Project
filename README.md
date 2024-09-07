# Vehicle Insurance Upselling Intelligence

## 1. Background

The vehicle insurance company X is facing challenges in increasing the conversion rate of customers interested in purchasing vehicle insurance. Although the company has data on customer profiles, many potential customers do not show interest or do not proceed with the purchase. Currently, only 12.3% of customers respond positively to insurance offers.

According to the LocaliQ 2023 benchmark report, the average conversion rate across all industries is around 7.04%. While a 12% conversion rate is relatively good, the cost of achieving this percentage is significant, with over 300,000 customers rejecting the insurance offer, resulting in wasted time and resources.

## 2. Problem Scope

### Goal:
To enhance the effectiveness of vehicle insurance sales.

### Objectives:
- Develop a machine learning model to predict customer interest in vehicle insurance and identify factors that influence this interest for segmentation purposes.
- Implement services based on customer segmentation.
- Increase sales and reduce associated costs.

### Business Metrics:
- **Conversion Rate:** Increase the precision of predicting customer interest.
- **Cost Efficiency:** Minimize wasted costs by accurately predicting True Negatives and reducing False Positives.

## 3. Data and Assumptions

- The raw dataset consists of 400,164 rows and 12 columns, including `id`, `gender`, `age`, `driving_license`, `region_code`, `previously_insured`, `vehicle_age`, `vehicle_damage`, `annual_premium`, `policy_sales_channel`, `vintage`, and `response`.
- The target column is `response`.

### About Dataset Columns

- **Gender:** The gender of the customer, which can be either "Male" or "Female."
- **Age:** The age of the customer, represented as a numerical variable.
- **Driving_License:** A binary variable indicating whether the customer has a valid driver's license or not.
- **Region_Code:** A unique region code for the customer, used to identify the customer's location.
- **Previously_Insured:** A binary variable indicating whether the customer already has vehicle insurance or not.
- **Vehicle_Age:** The age of the customer's vehicle, categorized as "1-2 Years," "< 1 Year," or "> 2 Years."
- **Vehicle_Damage:** A binary variable indicating whether the customer's vehicle has previously been damaged or not.
- **Annual_Premium:** The amount of premium the customer is required to pay annually.
- **Policy_Sales_Channel:** An anonymized code representing the communication channel through which the policy was sold to the customer.
- **Vintage:** The number of days the customer has been with the company.
- **Response:** The target variable indicating whether the customer is interested in vehicle insurance (1 for interested, 0 for not interested).

### Data Cleansing (Pre-EDA):
- **Missing Values:** Found in 5 columns: `id`, `gender`, `previously_insured`, `vehicle_damage`, and `policy_sales_channel`.
  - **Handling:** Missing `id` values were filled by continuing the existing sequence, `gender` was filled with the mode value, and the other columns were dropped.
- **Duplicate Data:** 18,076 duplicates were identified and removed.
- **Label Encoding:** Applied to `gender` and `vehicle_damage`.
- **One-Hot Encoding:** Applied to `vehicle_age`.
- The clean dataset consists of 378,136 rows with 14 columns.

## 4. Data Analysis

### Business Questions:
1. Do more positive responses come from customers who previously had vehicle insurance or those who did not?
2. Does customer age influence their response to insurance offers?
3. What is the age range of customers who give the most positive responses?
4. Does vehicle damage influence customer response to insurance offers?
5. Does vehicle age affect customer response to insurance offers?

### EDA:
- **Response by Previous Insurance Status**  
  > 99.6% of positive responses came from customers who do not currently have vehicle insurance.

- **Response by Age Group**  
  > 34.9% of positive responses were from customers aged 40-49.

- **Response by Vehicle Damage History**  
  > 97.8% of positive responses were from customers whose vehicles had previously experienced damage.

- **Response by Vehicle Age**  
  > 74.5% of positive responses were from customers with vehicles aged 1 to 2 years.

## 5. Data Preprocessing

### Data Preparation:
- After data cleansing, feature engineering was performed to create new features: `previously_claim_count`, `ownership_duration`, `policy_sales_channel`, and `customer_engagement_index`.

### Feature Selection:
- Used Chi-Square test for categorical data.
- Avoided features with significant outliers.
- **Selected Features:** `Age`, `Region_Code`,`Previously_Insured`, `Gender_Label`, `Vehicle_Damage_Label`, `Vehicle_Age_1-2 Year`, `Vehicle_Age_< 1 Year`, `Vehicle_Age_> 2 Years`,`Previous_Claims_Count`,`Ownership_Duration`

### Data Splitting:
- The dataset was split into 80% training data and 20% test data.

### Standardization:
- Applied to both training and test data.

### Handling Imbalanced Data:
- Used SMOTE on the training data to handle imbalanced classes.

## 6. Machine Learning

### Models Used:
- KNN, Naive Bayes, Random Forest, Decision Tree, XGBoost, Gradient Boosting, and Logistic Regression.
- Performed cross-validation and hyperparameter tuning to improve results.

### Model Performance:
- *Model performance table:*

| Model               | Precision (Train) | Precision (Test) |
|---------------------|----------------------|-----------------|
| KNN        | 0.88                 | 0.82            |                 
| Naive Bayes       | 0.90                 | 0.90            |                 
| Random Forest | 0.97                 | 0.82            |               
| Decision Tree       | 0.97                 | 0.82            |               
| XGBoost       | 0.87                 | 0.83            |               
| Gradient Boosting | 0.77                 | 0.77            |               
| Logistic Regression | 0.77                 | 0.77            |    

- *Confusion Matrix Naive Bayes:*

|                     | Predicted Negative| Predicted Positive|
|---------------------|----------------------|-----------------|
|  Actual Negative    | 43150 (58.64%) [TN]      | 21318 (28.97%) [FP]|                 
| Actual Positive     | 793 (1.08%) [FN]         | 8320 (11.31%) [TP]          | 

- *Calculate conversion rate (precision) from Confusion Matrix*
 Precision = TP / (TP + FP) = 8320 / (8320 + 21318) = 28.07%
 >(Up by 15.69% from previous conversion rate before model [12.3%])



## 7. Conclusion

- The selected model is Naive Bayes, with a weighted precision of 90% on both the test and training data.And it provides an increase of 15.69% in the conversion rate value, so that the conversion rate becomes 28.07%
- **Feature Importance:** The most important features identified are `previously_insured`, `vehicle_damage`, `vehicle_age < 1 year`, `age`, `vehicle_age 1-2 years`, `ownership_duration`, `previously_claim_count`, `region_code`, `gender`, and `vehicle_age > 2 years`.

## 8. Recommendations

Based on insights from EDA and the feature importance from machine learning, the following customer segments are recommended for focused insurance offerings:
- Customers with vehicles aged 1-2 years.
- Customers aged 40-49 years.
- Customers whose vehicles have experienced damage.
- Customers who do not currently have vehicle insurance.

### Future Development:
- Try to use feature importance only and develop this model into software with a user-friendly UI.

## 9. Usage/Installation

- Tools: Jupyter Notebook or Google Colab with Python 3.
- Library: `GaussianNB` (Naive Bayes, machine learning model) and `joblib` to load the model.
- Here is how to load and use the model:
```python
import joblib

# Load the model
model = joblib.load('nb_ins.pkl')

# Make predictions on new data
predictions = model.predict(X_new)
```

---

## Acknowledgments

I would like to express my sincere thanks to my fellow teammates in the **Data Wizard** group for their hard work and collaboration throughout this project. Your collective effort, dedication, and support made this project a success. I am grateful to have had the opportunity to work with such a talented and motivated team.
### Data Wizard:
* Dzulfikar Hanif Maulana 
* Abdul Hardia Amin
* Haerunnisa
* Nisrina Widya Nur Farhani (me)
