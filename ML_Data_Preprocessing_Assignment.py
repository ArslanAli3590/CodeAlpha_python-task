
# 📘 ML Assignment: Data Cleaning, Encoding, and Scaling in Python

## ✅ Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

## ✅ Step 2: Create a sample dataset with missing values and duplicates
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Alice'],  # duplicate row (Alice)
    'Age': [25, np.nan, 30, 22, 35, 25],  # missing value in Age
    'Gender': ['Female', 'Male', 'Male', np.nan, 'Female', 'Female'],  # missing in Gender
    'Salary': [50000, 60000, 55000, 52000, np.nan, 50000]  # missing in Salary
}
df = pd.DataFrame(data)

print("🔹 Original DataFrame")
print(df)
print("\n📊 Data Types:")
print(df.dtypes)
print("\n❓ Missing Values Count:")
print(df.isnull().sum())
print("\n🧠 Head of Data:")
print(df.head())

## 🧹 Step 3: Handle Missing Values and Remove Duplicates
# Fill missing numerical values with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# Fill missing categorical with mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

# Drop duplicates
df = df.drop_duplicates()

print("\n✅ After Cleaning (Missing + Duplicates):")
print(df)
print("\n📊 Data Types After Cleaning:")
print(df.dtypes)
print("\n❓ Missing Values After Cleaning:")
print(df.isnull().sum())

## 🔤 Step 4: Label Encoding (for 'Gender')
le = LabelEncoder()
df['Gender_LabelEncoded'] = le.fit_transform(df['Gender'])

print("\n🔤 After Label Encoding 'Gender':")
print(df[['Gender', 'Gender_LabelEncoded']])
print("\n📊 Data Types:")
print(df.dtypes)
print("\n🧠 Head:")
print(df.head())

## 🟩 Step 5: One-Hot Encoding (for 'Gender')
# One-hot encode and join back to df
df_onehot = pd.get_dummies(df, columns=['Gender'], prefix='Gender')

print("\n🟩 After One-Hot Encoding 'Gender':")
print(df_onehot.head())
print("\n📊 Data Types:")
print(df_onehot.dtypes)

## 📏 Step 6: Apply Min-Max Scaling and Standard Scaling
# Select numerical columns for scaling
num_cols = ['Age', 'Salary']

# Apply Min-Max Scaling
minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[num_cols] = minmax.fit_transform(df_minmax[num_cols])

print("\n📏 Min-Max Scaled Data:")
print(df_minmax.head())
print("\n📊 Data Types:")
print(df_minmax.dtypes)

# Apply Standard Scaling
standard = StandardScaler()
df_standard = df.copy()
df_standard[num_cols] = standard.fit_transform(df_standard[num_cols])

print("\n📐 Standard Scaled Data:")
print(df_standard.head())
print("\n📊 Data Types:")
print(df_standard.dtypes)
