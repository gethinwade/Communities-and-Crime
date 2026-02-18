# Package Imports
import re 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer 
from sklearn.preprocessing import StandardScaler


# Question 1
# ==================
# Importing the data
# ==================

with open('communities.names', 'r') as f:
    col_names = re.findall(r'^@attribute\s+(\S+)', f.read(), re.MULTILINE)

df = pd.read_csv('communities.data', header=None, names=col_names, na_values='?')

print(df.shape)


# Part A (i)
# ===========================================================
# Splitting the data 60/20/20 into train/validation/test sets
# ===========================================================

# Splitting the data
train, temp = train_test_split(df, test_size=0.4, random_state=42)

# Splitting the temp set in half to form val & test
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"Train Shape: {train.shape}")
print(f"Val Shape: {val.shape}")
print(f"Test Shape: {test.shape}")


# Part A (ii)
# ======================================================================
# Dropping non-predictive features and features with many missing values
# ======================================================================

print(train.dtypes.value_counts())

# Non-predictive features
to_drop = ['state', 'county', 'community', 'communityname', 'fold']

# 'Missing' threshold for dropping features is 80%
missing_pct = train.isnull().mean()
threshold = missing_pct[missing_pct > 0.8].index.tolist()

total_drop = list(set(to_drop + threshold))

train = train.drop(columns=total_drop)
val = val.drop(columns=total_drop)
test = test.drop(columns=total_drop)

print(f"Train shape: {train.shape}")


# Part A (iii)
# ==========================
# KNN Imputation where K = 5
# ==========================

# Defaults to K = 5
imputer = KNNImputer()

imputed_train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns, index=train.index) # Fit AND Transform
imputed_val = pd.DataFrame(imputer.transform(val), columns=val.columns, index=val.index) # Transform
imputed_test = pd.DataFrame(imputer.transform(test), columns=test.columns, index=test.index) # Transform

# Check for NaNs across rows & columns
print(imputed_train.isnull().sum().sum())
print(imputed_val.isnull().sum().sum())
print(imputed_test.isnull().sum().sum())


# Part A (iv)
# ===========================
# Standardization of Features
# ===========================

x_train = imputed_train.drop(columns='ViolentCrimesPerPop')
y_train = imputed_train['ViolentCrimesPerPop']

x_val = imputed_val.drop(columns='ViolentCrimesPerPop')
y_val = imputed_val['ViolentCrimesPerPop']

x_test = imputed_test.drop(columns='ViolentCrimesPerPop')
y_test = imputed_test['ViolentCrimesPerPop']

scaler = StandardScaler()
scaled_x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index) # Fit AND Transform
scaled_x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns, index=x_val.index) # Transform
scaled_x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index) # Transform


