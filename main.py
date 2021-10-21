# Libraries used

from itertools import groupby
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import PredictionError
from collections import Counter

# os.chdir('M:/project/git-repo/imdb-rating/')
df = pd.read_csv('./movies.csv')

# ==========================================
#                   EDA
# ==========================================
print(f"This data has {len(df)} rows, and {len(df.columns)} colums as following:\n{df.columns.values}\n")
print(" Data types ".center(30,'='),"\n", df.dtypes, "\n")

# Explore missing values
dfcols = list(df.columns.values)
print(" Missing Values ".center(30,'='))
for c in dfcols:
    print(f"{c.ljust(15)} {df[c].isna().sum()}")
# Convert data type from object to category
df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
df['year'] = df['year'].astype('category')

# Plot
# sns.set_theme(style="whitegrid")
# sns.boxplot(x=df['gross'])
# sns.displot(df, x="gross")
# sns.displot(df, x="budget",binwidth=50, hue="company")

df.select_dtypes(['float64']) # select specific types
df.select_dtypes(['category'])
df[['score','rating']].groupby('rating').describe()

# ==========================================
#            Data Preparation
# ==========================================
# To do
# drop NA of released, socre, vote, writer, star, country, company, runtime
# decide whether shoud we delete budget columns or not
# check outlier of numerical values
# impute rating colums (maybe based on mode)

#linear
df_clean = df.drop(['budget'], axis=1)
print('\ndrop budget')
print(f"This data has {len(df_clean)} rows, and {len(df_clean.columns)} colums as following:\n{df_clean.columns.values}")
print(f"Remain: {len(df_clean.columns)} columns\n")

# merge years into 5 years per group
print("group years in 5 years per group\n")
fiveyearly = ["{0} - {1}".format(i, i+4) for i in range(1980, 2025, 5)]
df_clean['year'] = pd.cut(df_clean['year'], range(1980, 2026,5), right=False, labels = fiveyearly)

# Explore missing values
dfcols = list(df_clean.columns.values)
print(" Missing Values ".center(30,'='))
for c in dfcols:
    print(f"{c.ljust(15)} {df_clean[c].isna().sum()}")

# Remove rows having more than 1 missing value (about 29 rows)
print('\ndrop rows that have more than 1 missing value')
df_clean.dropna(thresh=13, inplace=True)
print(f"Remain: {len(df_clean)} rows\n")

# Remove a few remaining missing value  (about 13 rows)
print('drop rows that have missing runtime, writer, or company')
df_clean.dropna(subset=['runtime','writer','company'], inplace=True)
print(f"Remain: {len(df_clean)} rows\n")

# Remove some levels of rating columns
print('drop rows that are TV Programs')
df_clean['rating'].value_counts()
rmvalue = ['Approved','TV-14','X','TV-PG','TV-MA']
for val in rmvalue:
    df_clean.drop(df_clean[df_clean.rating == val].index, axis=0, inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
df_clean['rating'] = df_clean['rating'].cat.remove_unused_categories()
print(f"Remain: {len(df_clean)} rows\n")

# merge Unrated and Not Rated
print('merge Unrated and Not Rated\n')
df_clean['rating'] = df_clean['rating'].replace({'Not Rated':'Unrated'})

# Remove unimportant features
print('drop unimportant features (name, released, writer, star, director)')
df_clean.columns
df_clean.drop(['name','released','writer','star','director'], axis=1, inplace=True)
print(f"Remain: {len(df_clean)} rows\n")

# Manipulate missing values
print('Manipulate missing values')
df_clean['gross'].describe()
df_clean['gross'].fillna(value=df_clean['gross'].median(), inplace=True)
df_clean.dropna(subset=['rating'], inplace=True)
df_clean.reset_index(drop=True, inplace=True)
print(f"Remain: {len(df_clean)} rows\n")

# Cannot put string type into model, try factorization or encoding before modeling
# Factorized categoorical features
"""df_clean.select_dtypes('category').columns
cate_col = ['rating' ,'genre', 'year', 'country', 'company']
for col in cate_col:
    factor = df_clean[col].factorize()
    df_clean[col] = factor[0]
    df_clean[col] = df_clean[col].astype('category') 
    del factor
del cate_col
df_clean.reset_index(drop=True, inplace=True)"""

# companies that have movies more than 10
print('drop companies that have movies less than 10')
big_companies = df_clean.company.value_counts()
big_companies = big_companies[big_companies >= 10]
# df_clean = df_clean[df_clean.groupby('company')['company'].transform('count').ge(10)]
df_clean.drop(df_clean[-df_clean['company'].isin(big_companies.index)].index, inplace=True)
df_clean.reset_index(drop=True, inplace=True)
df_clean['company'] = df_clean['company'].cat.remove_unused_categories()
print(f"Remain: {len(df_clean)} rows\n")

print(f"Remain: {len(df_clean)} rows from {len(df)} rows\n")
print("Final Missing Values ".center(30,'='))
for c in df_clean.columns:
    print(f"{c.ljust(15)} {df_clean[c].isna().sum()}")

df_clean = pd.get_dummies(df_clean)

df_clean['score'] = np.exp(df_clean['score'])

# ==========================================
#                 Modeling
# ==========================================
# Fit regression model
# Decision Tree
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
# x.shape = (row, col) and y = (row,): both are array
# dt = DecisionTreeRegressor(max_depth=2)
# dt.fit(train_x, train_y)
# yhat = dt.predict(test_x)

# Train/test split
X = df_clean.drop(['score'], axis=1)
Y = df_clean['score']
#trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.7, random_state=123)
trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.7, random_state=123)

print('\nmodeling phrase\n')

# Use DecisionTree
print('DecisionTree')
mse_list = []
# vary maxdepth from 1 to 20
for ndepth in range(1,21):
    model = DecisionTreeRegressor(random_state=1, max_depth=ndepth)
    model.fit(trainX, trainY)
    mse_list.append(mse(testY, model.predict(testX)))
mse_list = np.array(mse_list)
predicted_DT = model.predict(testX)
print(f"max_depth = {np.argsort(mse_list)[0]+1}, MSE = {mse_list[np.argsort(mse_list)[0]]:.4f}")
print('RMSE =', mse(testY, predicted_DT, squared=False))
print("R2 =", r2(testY,predicted_DT), "\n")

# Use GridSearchCV
print('DecisionTree with GridSearchCV')
model = DecisionTreeRegressor(random_state=1)
grid_model = GridSearchCV(model,
                  param_grid = {'max_depth': range(1, 21)},
                  cv=10,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')
grid_model.fit(trainX, trainY)
print(f"GridSearchCV, best max_depth = {list(grid_model.best_params_.values())[0]}, 10-fold CV MSE = {-grid_model.best_score_:.4f}")
print(f"max_depth = {list(grid_model.best_params_.values())[0]}, MSE = {mse(testY, grid_model.predict(testX)):.4f}")
predicted_DTwGSCV = grid_model.predict(testX)
print('RMSE =', mse(testY, predicted_DTwGSCV, squared=False))
print("R2 =", r2(testY,predicted_DTwGSCV), "\n")

print('Linear Regression')
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(trainX, trainY)
predicted = linear_model.predict(testX)
print('RMSE =', mse(testY, predicted, squared=False))
print('MSE =', mse(testY, predicted))
print('R2 =', r2(testY, predicted))

print('\nLinear Regression with 10-fold Cross Validation on training set only')
# Linear Regression
linear_model = LinearRegression()
crossscore = cross_val_score(linear_model, trainX, trainY, cv=10, scoring="r2")
print("R2 =", mean(crossscore))

# Visualization
# Instantiate the linear model and visualizer
visualizer = PredictionError(linear_model)
visualizer.fit(trainX, trainY)  # Fit the training data to the visualizer
visualizer.score(testX, testY)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure
