# London House Price Prediction – Kaggle ML Project

## Overview

This project aims to predict residential property prices in London using machine learning techniques.  
The dataset is provided through the Kaggle competition:

**London House Price Prediction: Advanced Techniques**  
https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques

The objective is to minimize **Mean Absolute Error (MAE)** between predicted and actual prices.

---

## Dataset

The dataset consists of property transaction records in London:

- **Training set:** 266,325 rows, 17 features  
- **Test set:** 16,547 rows, 16 features  
- **Target variable:** `price` (GBP)

### Key Features

- Geographic: `postcode`, `outcode`, `latitude`, `longitude`
- Structural: `bedrooms`, `bathrooms`, `floorAreaSqM`, `livingRooms`
- Property information: `propertyType`, `tenure`
- Energy rating: `currentEnergyRating`
- Temporal: `SaleMonth`, `SaleYear`

---

## Evaluation Metric

The competition metric is **Mean Absolute Error (MAE)**:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

---

## Data Preprocessing

### Handling Missing Values
- Imputed `floorAreaSqM` using median grouped by `propertyType`.
- Trained a small Linear Regression model to estimate missing `bedrooms` and `bathrooms` based on floor area.

### Outlier Treatment
- Applied log transformation to reduce right-skewness of the target variable:
$$
y = \log(1 + price)
$$

### Postcode Decomposition
- Split `postcode` into:
  - `outcode` (geographic area)
  - `incode` (local subdivision)

---

## Exploratory Data Analysis (EDA)

Key findings:

- Price distribution is highly right-skewed; log transformation improves normality.
- Detached and Semi-Detached houses have the highest average prices.
- Strong positive correlation between floor area and price.

---

## Feature Engineering

Additional features were created:

- **Geographic features:** `outcode`, `incode`
- **Structural feature:** total rooms = bedrooms + bathrooms + 1
- **Temporal trend feature:**  
  `trend = 12 × (SaleYear - 1995) + SaleMonth`  
  and its quadratic term
- **Target Encoding:** applied to `outcode` and `propertyType`
- **Interaction feature:** floor area × total rooms

---

## Models Evaluated

The following regression models were trained and compared:

1. Ridge Regression (L2 regularization)
2. Lasso Regression (L1 regularization)
3. LightGBM
4. XGBoost

### Hyperparameter Tuning

Used **5-fold cross-validation** with GridSearchCV.

Example search space:

- Ridge / Lasso: α ∈ {0.1, 1, 10, 100}
- LightGBM:
  - n_estimators: {500, 600}
  - max_depth: {7, 9}
  - learning_rate: {0.5, 0.6, 0.7}
- XGBoost:
  - n_estimators: {500, 600}
  - max_depth: {7, 9, 10}
  - learning_rate: {0.5, 0.6, 0.7}

---

## Results

| Model     | MAE        | RMSE      | R²     |
|------------|------------|-----------|--------|
| Ridge      | 366,513    | 31,610,230| -713   |
| Lasso      | 398,028    | 1,206,240 | -0.04  |
| LightGBM   | 112,631    | 504,080   | 0.82   |
| XGBoost    | 79,401     | 307,817   | 0.93   |

**Best Model:** XGBoost  
R² ≈ 0.93, significantly outperforming linear baselines.

---

## Conclusion

- Gradient boosting models (XGBoost, LightGBM) significantly outperform linear regression models.
- Log transformation and feature engineering played a crucial role in improving performance.
- Further improvements could include:
  - Stacking ensembles
  - Bayesian hyperparameter optimization
  - Incorporating external spatial data

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- Matplotlib / Seaborn

---

## 🏆 Final Competition Result

As of **July 2025**, this solution achieved:

🥇 **Top 1 position on the Kaggle public leaderboard**

The final model (XGBoost with tuned hyperparameters and engineered features) achieved the best performance among all submissions at that time.

### Leaderboard Screenshot

![Kaggle Leaderboard Result](image.png)

*Leaderboard ranking captured in July 2025.*

## Author

Dang Quoc Cuong, Pham Huynh Long Vu  
University of Information Technology (UIT), VNU-HCM
