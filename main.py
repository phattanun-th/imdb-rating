# Libraries used
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# os.chdir('M:/project/git-repo/imdb-rating/')
df = pd.read_csv('./movies.csv')

# ==========================================
#                   EDA
# ==========================================
print(f"This data has {len(df)} rows, and {len(df.columns)} colums as following:")
print(df.columns.values)
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
sns.boxplot(x=df['budget'])
# sns.displot(df, x="budget", binwidth = 10)
# sns.displot(df, x="budget",binwidth=50, hue="company")
df['rating'].value_counts() 
df['company'].value_counts() # 2385 companies
df.select_dtypes(['float64']) # select specific types
df.select_dtypes(['category'])

# To do
# drop NA of released, socre, vote, writer, star, country, company, runtime
# decide whether shoud we delete budget columns or not
# check outlier of numerical values
# impute rating colums (maybe based on mode)

#linear