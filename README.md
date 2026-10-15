# Abalone Age Prediction: Statistical vs Algorithmic Models

Predicting the age of abalone from physical measurements, and comparing a statistical model (polynomial regression with Lasso) with an algorithmic one (XGBoost).
MSc Data Science coursework (Data Mining), Lancaster University Leipzig.

Age is normally measured by cutting the shell, staining it and counting rings under a microscope, which is slow and expensive. Both models estimate it from length, diameter, height and weight measurements instead.

## Approach

- **Data:** 4,000 abalone with sex, length, diameter, height and four weight measurements. Age = rings + 1.5.
- **Feature engineering:** biologically motivated ratios such as meat ratio and viscera ratio.
- **Polynomial regression:** degree-2 expansion to 90 terms, then LassoCV (10-fold) kept 26 of them.
- **XGBoost:** hyperparameters tuned with Hyperopt (TPE) during development, then fixed for the final evaluation.
- **Validation:** nested k-fold cross-validation with min-max scaling fitted on training data only, then a held-out test set.
- **Interpretation:** SHAP values. Shell weight is the strongest predictor.

## Results (held-out test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | 0.0550 (about 1.54 years) | 0.0772 | 0.6151 |
| Polynomial regression + Lasso | 0.0552 | 0.0779 | 0.6081 |

XGBoost is slightly better on every metric. The polynomial model stays a clear, interpretable baseline. Both models make larger errors on older abalone.

## Run it

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn scipy
python 39991938_XGBoost_model.py
python 39991938_Pol_Reg_Model.py
```

The scripts read the dataset `abalone.head` from the current folder. The path is written Windows-style (`\\abalone.head`), so on Mac or Linux change it to `/abalone.head`. The full write-up is `39991938_Data_Mining_Coursework.pdf`.
