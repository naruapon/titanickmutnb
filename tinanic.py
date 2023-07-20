import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('test.csv')

# Data exploration and preprocessing (if required)
# ... (you can perform some data cleaning and processing here if needed)

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
