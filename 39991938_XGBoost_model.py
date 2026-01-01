#This code is the result of the Data Mining Coursework Task
#The Code is dervied and implemented as a result of independent research that resulted in the building and implementation of two models

#The following code is for our XGBOOST Model


## Library Implementation Stage
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap



## Importing Stage
DS_file = os.getcwd() + '\\abalone.head'                                #finding the file named 'abalone.head' in our zip folder
abalone_DS = pd.read_csv(DS_file, sep=',', header=None)                 #loading the file in a variable


## Preprocessing Stage

#Naming the columns
abalone_DS.columns = ['sex', 'length', 'diameter','height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'number_of_rings']

#Calculating and adding 'age' as a new colum + removing the previous 'number_of_rings' column
abalone_DS['age'] = abalone_DS['number_of_rings'] + 1.5                 #new age column added
abalone_DS.drop('number_of_rings', axis = 1, inplace = True)            #number_of_rings column removed

#Replacing Missing Values | Method of choice: removing rows that contain missing values | Possible missing values are converted to 'np.nan' and the rows that contain 'np.nan' are removed

missing_values = [                                                      #creating a variable that will hold all possible types of missing values
    r'^\s*$',                                                           #empty or whitespace-only cells
    'NA', 'NaN', 'null', 'None',                                        #in cases where any of these words appear in the observations
    r'(?i)^(?![mfi]$).*'                                                #any string that is not m, f, or i (case-insensitive)
]

abalone_DS.replace(missing_values, np.nan, regex=True, inplace=True)    #.replace pandas function is used to replace all possible 'missing_values' with np.nan
abalone_DS.dropna(inplace=True)                                         #.dropna is used to recognize all 'np.nan' values and to remove all the rows that contain those values

#Encoding Values | 'sex' column is split into 3 hot encoded 'male','female' and 'infant' columns
encoded_features = pd.get_dummies(abalone_DS['sex'], dtype = int)       #.get_dummies pandas function is used to turns the 'sex' column into one column per unique value
encoded_features.columns = ['female', 'male', 'infant']                 # naming those generated columns

abalone_DS = pd.concat([encoded_features, abalone_DS], axis=1)          #adding the encoded features to the beginning of our current dataset
abalone_DS.drop('sex', axis = 1, inplace = True)                        #removing the 'sex' column as it's no longer needed

#Feature Selection & Engineering | Some columns we're removed. Some new engineered features were added. We also checked if we can remove any features based on varience level, but none was removed
abalone_DS.drop(['male', 'female'], axis = 1, inplace = True)           #removing the 'male' and 'female' columns


abalone_DS['meat_ratio']      = abalone_DS['shucked_weight'] / abalone_DS['whole_weight'] #Feature Engineering Process
abalone_DS['viscera_ratio']   = abalone_DS['viscera_weight'] / abalone_DS['whole_weight']
abalone_DS['shell_ratio']     = abalone_DS['shell_weight']   / abalone_DS['whole_weight']
abalone_DS['soft_weight']     = abalone_DS['shucked_weight'] + abalone_DS['viscera_weight']

abalone_DS.var(ddof=1)                                                  #checking variences of all columns, no sufficient reason for any column removal was found


## Data Splitting Stage | Method chosen: 80/20 Split 

X = abalone_DS.drop('age', axis = 1)                                    #initializing the independent variables | selected features
Y = abalone_DS['age']                                                   #initializing the dependent variables   | outcome


X_train, X_test, Y_train, Y_test = train_test_split(                    #creating the variables to hold the set split for the data
    X, Y, test_size = 0.20, random_state = 42                           #test size refers to the 20% size of the test we want to have | random_state is used to be stuck on the same random partion of the split
)

print("Train size:", X_train.shape)                                     #printing amount of observations for training set
print("Test size:", X_test.shape)                                       #printing amount of observations for test set

## Normalization on Training Set | Method chosen: min/max [0,1] Normalization. Normalization will be performed both on training and test set with only training set's min/max parameters

X_train_min = X_train.min()                                             #creating the training min/max values for X and Y variable of the training set                                          
X_train_max = X_train.max()

Y_train_min = Y_train.min()
Y_train_max = Y_train.max()

X_train = (X_train - X_train_min) / (X_train_max - X_train_min)         #normalizing the X and Y variables of the training set 
Y_train = (Y_train - Y_train_min) / (Y_train_max - Y_train_min)                                  

X_test = (X_test - X_train_min) / (X_train_max - X_train_min)           #normalizing the X and Y variables of the test set
Y_test = (Y_test - Y_train_min) / (Y_train_max - Y_train_min)



#### XGBoost Regression Model Stage

## Initializing Final XGBoost Model
#All hyperparameters used here were determined during development and are fixed for final evaluation.
#No hyperparameter tuning or optimisation takes place in this code.

final_xgb_model = XGBRegressor(
    n_estimators=520,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    n_jobs=1
)

final_xgb_model.fit(X_train, Y_train)                                   #training the final XGBoost model on the training set


Y_pred_norm = final_xgb_model.predict(X_test)                           #predicting normalized age values on the test set


## Printing Test Performance Stage
print("\nFinal Test Performance")
print("MAE :", mean_absolute_error(Y_test, Y_pred_norm))
print("RMSE:", mean_squared_error(Y_test, Y_pred_norm)**0.5)
print("R²  :", r2_score(Y_test, Y_pred_norm))


#Reverting predictions back to original age scale for interpretability
Y_test_orig = Y_test * (Y_train_max - Y_train_min) + Y_train_min
Y_pred_orig = Y_pred_norm * (Y_train_max - Y_train_min) + Y_train_min

print("\nFinal Test Performance (Original Age Scale)")
print("MAE :", mean_absolute_error(Y_test_orig, Y_pred_orig))
print("RMSE:", mean_squared_error(Y_test_orig, Y_pred_orig)**0.5)
print("R²  :", r2_score(Y_test_orig, Y_pred_orig))


## SHAP Interpretability Stage
#SHAP values are used to understand how each feature contributes to the final predictions

explainer = shap.TreeExplainer(final_xgb_model)                         #initializing SHAP TreeExplainer
shap_values = explainer.shap_values(X_train)                            #calculating SHAP values for training data

shap.summary_plot(shap_values, X_train)                                 #feature impact summary plot
shap.summary_plot(shap_values, X_train, plot_type='bar')                #mean absolute feature importance plot


## Visualisation Stage | Predicted vs Actual Graph
#We see how far off we are from the original output

y_pred_xgb = Y_pred_orig                                             #using predictions reverted back to original age scale

plt.figure(figsize=(8, 5))
plt.scatter(Y_test_orig, y_pred_xgb, alpha=0.6)

#building line that shows perfect prediction
plt.plot(
    [Y_test_orig.min(), Y_test_orig.max()],
    [Y_test_orig.min(), Y_test_orig.max()],
    'r--',
    label='Perfect Prediction'
)

plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Actual (XGBoost Regression)")
plt.legend()
plt.grid(False)
plt.show()
