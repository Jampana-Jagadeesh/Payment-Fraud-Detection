
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# Load your dataset
df = pd.read_csv('/content/drive/MyDrive/financial_anomaly_data.csv')

# Check for missing values
print(df.isnull().sum())

# Impute missing values (replace NaN with the mean)
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])

# Normalize data if needed
df[features] = (df[features] - df[features].mean()) / df[features].std()

# Train Isolation Forest model
model = IsolationForest(contamination=0.01)
model.fit(df[features])

# Predict outliers/fraudulent transactions
df['Fraudulent'] = model.predict(df[features])

# Identify potential fraudulent transactions
potential_fraud = df[df['Fraudulent'] == -1]
print(potential_fraud)



# Example feature engineering
df['TransactionHour'] = pd.to_datetime(df['Timestamp']).dt.hour



# Filter data for June
june_data = df[(df['Timestamp'] >= '2023-06-01') & (df['Timestamp'] < '2023-07-01')]

# Group by TransactionType and calculate spend
monthly_spend = june_data.groupby('TransactionType')['Amount'].sum()
print(monthly_spend)



import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split the data
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Train the model on the training set
model.fit(train[features])

# Evaluate on the test set
predictions = model.predict(test[features])

# Assess the performance
print(classification_report(test['Fraudulent'], predictions))


import matplotlib.pyplot as plt
import seaborn as sns

# Convert 'Timestamp' to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Assuming df is your dataframe with the 'Fraudulent' column and the anomaly scores
anomaly_scores = model.decision_function(df[features])

# Create a scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['Timestamp'], y=anomaly_scores, hue=df['Fraudulent'], palette={-1: 'red', 1: 'green'})
plt.title('Anomaly Scores Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Anomaly Score')
plt.legend(title='Fraudulent', loc='upper right', labels={-1: 'Potential Fraud', 1: 'Normal'})
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Create a histogram of anomaly scores
plt.figure(figsize=(10, 6))
sns.histplot(anomaly_scores, bins=50, kde=True, color='skyblue')
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Create a box plot of anomaly scores
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fraudulent'], y=anomaly_scores, palette={-1: 'red', 1: 'green'})
plt.title('Box Plot of Anomaly Scores for Normal and Fraudulent Transactions')
plt.xlabel('Transaction Class')
plt.ylabel('Anomaly Score')
plt.xticks(ticks=[0, 1], labels=['Normal', 'Potential Fraud'])
plt.tight_layout()
plt.show()


# Create a violin plot of anomaly scores
plt.figure(figsize=(10, 6))
sns.violinplot(x=df['Fraudulent'], y=anomaly_scores, palette={-1: 'red', 1: 'green'}, inner="quartile")
plt.title('Violin Plot of Anomaly Scores for Normal and Fraudulent Transactions')
plt.xlabel('Transaction Class')
plt.ylabel('Anomaly Score')
plt.xticks(ticks=[0, 1], labels=['Normal', 'Potential Fraud'])
plt.tight_layout()
plt.show()

# Assuming df is your dataframe with the 'Fraudulent' column and the anomaly scores
features = ['Amount']  # Adjust based on your features

# Predict outliers/fraudulent transactions and add AnomalyScore to the DataFrame
df['AnomalyScore'] = model.decision_function(df[features])

# Select relevant columns for the pair plot
pair_plot_cols = ['Amount', 'AnomalyScore', 'Fraudulent']

# Create a pair plot
plt.figure(figsize=(12, 8))
sns.pairplot(df[pair_plot_cols], hue='Fraudulent', palette={-1: 'red', 1: 'green'})
plt.suptitle('Pair Plot of Relevant Variables', y=1.02)
plt.show()

