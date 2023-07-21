import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Data exploration and preprocessing (if required)
# ... (you can perform some data cleaning and processing here if needed)
# Data cleaning and feature selection
selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']
df = df[selected_features]

# Handling missing values
# For 'Age' column, impute missing values with the median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# For 'Embarked' column, impute missing values with the most common value
most_common_embarked = df['Embarked'].mode().iloc[0]
df['Embarked'].fillna(most_common_embarked, inplace=True)

# Encoding categorical variables
df = pd.get_dummies(df, drop_first=True)

# Verify the updated dataframe
print(df.head())

# Data visualization
# Create a figure and set the size
plt.figure(figsize=(12, 8))

# Plot a histogram for age distribution
sns.histplot(data=df, x='Age', bins=20, kde=True)
plt.title('Age Distribution on Titanic')
plt.xlabel('Age')
plt.ylabel('Count')

# Display the plot
plt.show()

# Create a figure for passenger class distribution
plt.figure(figsize=(8, 6))

# Plot a countplot for passenger class distribution
sns.countplot(data=df, x='Pclass')
plt.title('Passenger Class Distribution on Titanic')
plt.xlabel('Passenger Class')
plt.ylabel('Count')

# Display the plot
plt.show()

# Create a figure for survival distribution
plt.figure(figsize=(8, 6))

# Plot a countplot for survival distribution
sns.countplot(data=df, x='Survived')
plt.title('Survival Distribution on Titanic')
plt.xlabel('Survived')
plt.ylabel('Count')

# Display the plot
plt.show()

# Create a figure for survival distribution based on passenger class
plt.figure(figsize=(8, 6))

# Plot a countplot for survival distribution based on passenger class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Distribution on Titanic based on Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')

# Display the plot
plt.show()
