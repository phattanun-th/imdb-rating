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
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.chdir('M:/project/git-repo/imdb-rating/')
df = pd.read_csv('./movies.csv')

# ==========================================
#                   EDA
# ==========================================
print(f"This data has {len(df)} rows, and {len(df.columns)} colums as following:\n{df.columns.values}")
print(" Data types ".center(30,'='),"\n", df.dtypes)

# Explore missing values
dfcols = list(df.columns.values)
print(" Missing Values ".center(30,'='))
for c in dfcols:
    print(f"{c.ljust(15)} {df[c].isna().sum()}")

# Convert data type from object to category
df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
df['year'] = df['year'].astype('category')

# Should we impute missing values?
sns.set_theme(style="whitegrid")
sns.boxplot(x=df_clean['gross'])
sns.displot(df_clean, x="gross")
# sns.displot(df, x="budget",binwidth=50, hue="company")

# Data transformation
df['rating'].value_counts() 
# df['company'].value_counts() # 2385 companies
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
df_clean = df.drop(['budget'], axis=1)
print(f"This data has {len(df_clean)} rows, and {len(df_clean.columns)} colums as following:\n{df_clean.columns.values}")
print(f"Remain: {len(df_clean.columns)} columns")
# Explore missing values
dfcols = list(df_clean.columns.values)
print(" Missing Values ".center(30,'='))
for c in dfcols:
    print(f"{c.ljust(15)} {df_clean[c].isna().sum()}")
# Remove rows having more than 1 missing value (about 29 rows)
df_clean.dropna(thresh=13, inplace=True)
# Remove a few remaining missing value  (about 13 rows)
df_clean.dropna(subset=['runtime','writer','company'], inplace=True)
# Remove some levels of rating colums
print(f"Remain: {len(df_clean)} rows")
df_clean['rating'].value_counts()
rmvalue = ['Approved','TV-14','X','TV-PG','TV-MA']
for val in rmvalue:
    df_clean.drop(df_clean[df_clean.rating == val].index, axis=0, inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
print(f"Remain: {len(df_clean)} rows")

# Manipulate missing values
df_clean['gross'].describe()
df_clean['gross'].fillna(value=df_clean['gross'].median(), inplace=True)
df_clean.dropna(subset=['rating'], inplace=True)
print(f"Remain: {len(df_clean)} rows from {len(df)} rows")

# df_clean.select_dtypes(['float64']) # select specific types
# df_clean.select_dtypes(['category'])

# Train/test split


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
