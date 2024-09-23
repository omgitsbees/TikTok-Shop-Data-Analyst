import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\kyleh\OneDrive\Desktop\PowerBI Dashboards\Customer Churn\customer_churn_dataset-training-master.csv'
data = pd.read_csv(file_path)

# Check for NaN values in the dataset
print("NaN values in 'Churn':", data['Churn'].isnull().sum())

# Drop rows with NaN values in 'Churn'
data = data.dropna(subset=['Churn'])

# Convert 'Last Interaction' to datetime format and calculate 'recency'
data['Last Interaction'] = pd.to_datetime(data['Last Interaction'])
data['recency'] = (pd.to_datetime('today') - data['Last Interaction']).dt.days

# Handle categorical data (example with 'Contract Length')
# Assuming 'Contract Length' has categorical values like 'Quarterly', 'Yearly', etc.
data = pd.get_dummies(data, columns=['Contract Length'], drop_first=True)

# Feature Selection
features = ['Age', 'Payment Delay', 'Support Calls', 'Tenure', 'Total Spend', 'Usage Frequency', 'recency']

# Add the newly created dummy columns to the features list
dummy_columns = [col for col in data.columns if col.startswith('Contract Length_')]
features.extend(dummy_columns)

# Define the target variable 'Churn'
target = 'Churn'

# Split the dataset into features (X) and target (y)
X = data[features]
y = data[target]

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f'Accuracy: {accuracy:.2f}\n')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)

# Plot the Confusion Matrix for better visualization
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
