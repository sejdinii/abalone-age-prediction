# Abalone Age Prediction - Comparative ML Study

Predicting abalone age from morphological measurements using statistical and algorithmic approaches.

## 📊 Project Overview
- **Objective**: Compare Polynomial Regression (with Lasso) vs XGBoost for age prediction
- **Dataset**: 4,000 abalone samples with 9 morphological features
- **Best Model**: XGBoost achieved R² = 0.615, MAE = 1.54 years

## 🔧 Technologies Used
- Python (Pandas, NumPy, scikit-learn)
- XGBoost, Hyperopt
- SHAP for model interpretability
- Matplotlib, Seaborn for visualization

## 📈 Key Results
- Engineered biologically-motivated features (meat ratio, viscera ratio)
- Applied Lasso regularization (reduced 90 terms to 26)
- Implemented nested k-fold cross-validation
- SHAP analysis identified shell weight and meat ratio as top predictors

## 🎓 Context
University coursework project (2024-2025)

## 📝 Note
This project demonstrates comparative analysis of statistical and algorithmic machine learning approaches for regression problems.
