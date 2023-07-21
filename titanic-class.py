import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Data preprocessing
# Select relevant features
selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Survived']
df = df[selected_features].dropna()

# Convert 'Sex' to binary representation (0 for female, 1 for male)
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# Split the data into features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')

# Data visualization
# Visualize the distribution of survived vs. not survived passengers
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='pastel')
plt.title('Survival Distribution')
plt.xlabel('Survived (0 = Not Survived, 1 = Survived)')
plt.ylabel('Count')
plt.show()

# Visualize the survival rate based on Pclass
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df, palette='muted')
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.show()

# Visualize the survival rate based on Sex
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df, palette='dark')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Survival Rate')
plt.show()

# Visualize the survival rate based on Age
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', bins=20, kde=True, palette='coolwarm')
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
