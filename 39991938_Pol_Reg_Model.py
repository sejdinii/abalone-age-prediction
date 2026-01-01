#This code is the result of the Data Mining Coursework Task
#The Code is dervied and implemented as a result of independent research that resulted in the building and implementation of two models

#The following code is for our Polynomial Regression Model


## Library Implementation Stage
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




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



#### Polynomial Regressiong Model Stage

## Initializing our Degree-2 Polynomial Regression

pol_degree = PolynomialFeatures(degree=2, include_bias=False)     #This only sets up the idea of making degree-2 features. No features are made yet until we run fit_transform.

training_set_pol = pol_degree.fit_transform(X_train)              #creates new polynomial features from X_train (training set) based on the degree-2 rule
                                                                  #fit is used here so pol_degree learns how to build these degree-2 features from the training data

test_set_pol = pol_degree.transform(X_test)                       #pol_degree already learned the feature creation rules from training, 
                                                                  #so here we only use transform to create the same type of features for the test set (no learning on test)

print("Total polynomial features:", training_set_pol.shape[1])    #just for interpretation purposes, we print the initial features created
## Selecting ONLY the 26 features chosen during development



lasso_selected_features = [                                       #In our previous code, we ran an extensive Lasso Regularization process and the selected_terms were the chosen features by Lasso Regularization
    'infant', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
    'shell_weight', 'viscera_ratio', 'infant^2', 'infant length', 'infant diameter',
    'infant height', 'infant viscera_weight', 'infant shell_weight', 'infant viscera_ratio',
    'length^2', 'length viscera_ratio', 'diameter^2', 'height^2', 'whole_weight^2',
    'shucked_weight^2', 'shucked_weight shell_weight', 'shell_weight soft_weight',
    'meat_ratio^2', 'viscera_ratio^2', 'shell_ratio^2'
]


all_polynomial_features = pol_degree.get_feature_names_out(X_train.columns)#we create a variable that will store all the names of the original features created by training_set_pol and test_pol ie. our polynomial regression initially


X_train_pol_df = pd.DataFrame(training_set_pol, columns=all_polynomial_features)#Initially our 90 created features were matrices. Matrices dont have column names so we cannot filter by feature name directly. To filter by feature name we convert the polynomial arrays to DataFrames.
X_test_pol_df  = pd.DataFrame(test_set_pol,  columns=all_polynomial_features)


X_train_selected = X_train_pol_df[lasso_selected_features]          #Keeping only the 26 selected polynomial terms for both on the training set and test set
X_test_selected  = X_test_pol_df[lasso_selected_features]

print("Polynomial features kept:", X_train_selected.shape[1])


## Final Model Creation Stage (Training LASSO with fixed alpha weights)
#Apart from selecting features Lasso helps us prevent overfitting and increase performance by determining the weight of the selected features as well, that's why Alpha is kept to train our final model.


alpha_value = 1.3041184652925738e-05                                #This alpha was discovered during development. No tuning happens here.
final_polynomial_reg = Lasso(alpha=alpha_value, max_iter=200000)


final_polynomial_reg.fit(X_train_selected, Y_train)                 #Training the model on the selected polynomial features


Y_pred_norm = final_polynomial_reg.predict(X_test_selected)         #Predicting on the test set


## Printing Test Performance Stage
print("\nFinal Test Performance")
print("MAE :", mean_absolute_error(Y_test, Y_pred_norm))
print("RMSE:", mean_squared_error(Y_test, Y_pred_norm)**0.5)
print("R²  :", r2_score(Y_test, Y_pred_norm))


#Reverting predictions back to original age scale. We do this simply to have graphs that represent closer real life values
Y_test_orig = Y_test * (Y_train_max - Y_train_min) + Y_train_min
Y_pred_orig = Y_pred_norm * (Y_train_max - Y_train_min) + Y_train_min

print("\nFinal Test Performance (Original Age Scale)")
print("MAE :", mean_absolute_error(Y_test_orig, Y_pred_orig))
print("RMSE:", mean_squared_error(Y_test_orig, Y_pred_orig)**0.5)
print("R²  :", r2_score(Y_test_orig, Y_pred_orig))


## Plotting Stage | all plots now use original age values so that interpretations can be close to real Abalone values


residuals_orig = Y_test_orig - Y_pred_orig                          #Calculating residuals in original scale (actual - predicted)

#Predicted vs Actual Values Plot. We see how far off we are from the original output
plt.figure(figsize=(7,5))
plt.scatter(Y_test_orig, Y_pred_orig, alpha=0.6)                    #scatter plot of actual vs predicted age
plt.plot([Y_test_orig.min(), Y_test_orig.max()],
         [Y_test_orig.min(), Y_test_orig.max()],
         'r--')                                                     #building line that shows perfect prediction
plt.xlabel("Actual Age (years)")
plt.ylabel("Predicted Age (years)")
plt.title("Predicted vs Actual (Polynomial Regression)")
plt.show()

#Residuals vs Fitted. We see how the numerical difference of our predictions vs actual outcomes
plt.figure(figsize=(7,5))
plt.scatter(Y_pred_orig, residuals_orig, alpha=0.6)          #scatter plot of predicted vs model errors
plt.axhline(0, color='red', linestyle='--')                  #line showing where residuals = 0
plt.xlabel("Predicted Age (years)")
plt.ylabel("Residuals (years)")
plt.title("Residuals vs Fitted")
plt.show()

#Residual distribution. shows how big the model’s mistakes are and if it mostly guesses too high or too low
plt.figure(figsize=(7,5))
plt.hist(residuals_orig, bins=30, edgecolor='black')         #histogram of model errors
plt.xlabel("Residual (years)")
plt.title("Residual Distribution")
plt.show()

#Q-Q Plot. checks if our errors follow a normal pattern or if something looks off
plt.figure(figsize=(6,6))
stats.probplot(residuals_orig, dist="norm", plot=plt)        #checking if residuals follow a normal pattern
plt.title("Q-Q Plot of Residuals")
plt.show()
