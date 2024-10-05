import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import pickle
import time
import warnings
warnings.filterwarnings('ignore')
import os
import json 

# Define the directory path
directory = r'C:\Users\kyleh\OneDrive\Desktop\PowerBI Dashboards\TikTok Shop User Recommendations'

# Load the datasets
user_interactions = pd.read_csv(os.path.join(directory, 'user_interactions.csv'))
product_info = pd.read_csv(os.path.join(directory, 'product_info.csv'))
sales_data = pd.read_csv(os.path.join(directory, 'sales_data.csv'))

# Define a function to parse the 'Amount' column
def parse_amount(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    else:
        return x

# Parse the 'Amount' column
sales_data['Amount'] = sales_data['Amount'].apply(parse_amount)

# Define a function to extract the 'Price' values
def extract_price(x):
    if isinstance(x, list):
        return [item['Price'] for item in x]
    else:
        return x

# Extract the 'Price' values from each dictionary
sales_data['Price'] = sales_data['Amount'].apply(extract_price)

# Plot the trend of sales over time
plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Price', data=sales_data.explode('Price'))
plt.title('Trend of Sales over Time')
plt.show()

# Plot the distribution of Age
plt.figure(figsize=(10,6))
sns.histplot(user_interactions['Age'], bins=10)
plt.title('Distribution of Age')
plt.show()

# Plot the distribution of Annual Income
plt.figure(figsize=(10,6))
sns.histplot(user_interactions['Annual Income'], bins=10)
plt.title('Distribution of Annual Income')
plt.show()

# Plot the distribution of Gender
plt.figure(figsize=(10,6))
sns.countplot(x='Gender', data=user_interactions)
plt.title('Distribution of Gender')
plt.show()

# Plot the correlation between Purchase History and Browsing History
plt.figure(figsize=(10,6))
sns.heatmap(user_interactions[['Purchase History', 'Browsing History']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Purchase History and Browsing History')
plt.show()

# Plot the distribution of stars
plt.figure(figsize=(10,6))
sns.countplot(x='stars', data=product_info.dropna(subset=['stars']))
plt.title('Distribution of Stars')
plt.show()

# Plot the distribution of reviews
plt.figure(figsize=(10,6))
sns.countplot(x='reviews', data=product_info.dropna(subset=['reviews']))
plt.title('Distribution of Reviews')
plt.show()

# Plot the relationship between price and listPrice
plt.figure(figsize=(10,6))
sns.scatterplot(x='price', y='listPrice', data=product_info)
plt.title('Relationship between Price and List Price')
plt.show()

# Plot the distribution of category_id
plt.figure(figsize=(10,6))
sns.countplot(x='category_id', data=product_info.dropna(subset=['category_id']))
plt.title('Distribution of Category ID')
plt.show()

# Plot the trend of sales over time
plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Amount', data=sales_data.dropna(subset=['Date', 'Amount']))
plt.title('Trend of Sales over Time')
plt.show()

# Plot the distribution of Status
plt.figure(figsize=(10,6))
sns.countplot(x='Status', data=sales_data.dropna(subset=['Status']))
plt.title('Distribution of Status')
plt.show()

# Plot the distribution of Fulfilment
plt.figure(figsize=(10,6))
sns.countplot(x='Fulfilment', data=sales_data.dropna(subset=['Fulfilment']))
plt.title('Distribution of Fulfilment')
plt.show()

# Plot the correlation between Qty and Amount
plt.figure(figsize=(10,6))
sns.heatmap(sales_data[['Qty', 'Amount']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Qty and Amount')
plt.show()

# Print the first few rows of each dataset
print(user_interactions.head())
print(product_info.head())
print(sales_data.head())

# Print the column names of the user_interactions dataframe
print(user_interactions.columns)

# Plot the distribution of user interactions
plt.figure(figsize=(10,6))
sns.countplot(x='interaction_type', data=user_interactions)
plt.title('User Interaction Distribution')
plt.show()

# Plot the distribution of product categories
plt.figure(figsize=(10,6))
sns.countplot(x='category', data=product_info)
plt.title('Product Category Distribution')
plt.show()

# Plot the distribution of sales amounts
plt.figure(figsize=(10,6))
sns.distplot(sales_data['sales_amount'])
plt.title('Sales Amount Distribution')
plt.show()

# Plot the distribution of user interactions by product category
plt.figure(figsize=(10,6))
sns.countplot(x='category', hue='interaction_type', data=user_interactions.merge(product_info, on='product_id'))
plt.title('User Interaction Distribution by Product Category')
plt.show()

# Plot the distribution of sales amounts by product category
plt.figure(figsize=(10,6))
sns.boxplot(x='category', y='sales_amount', data=sales_data.merge(product_info, on='product_id'))
plt.title('Sales Amount Distribution by Product Category')
plt.show()

# Define a function to preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join the tokens back into a string
    text = ' '.join(tokens)
    return text

# Preprocess the text data
user_interactions['text'] = user_interactions['text'].apply(preprocess_text)
product_info['description'] = product_info['description'].apply(preprocess_text)

# Define a function to split the data into training and testing sets
def split_data(data, test_size):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

# Split the data into training and testing sets
train_user_interactions, test_user_interactions = split_data(user_interactions, test_size=0.2)
train_product_info, test_product_info = split_data(product_info, test_size=0.2)
train_sales_data, test_sales_data = split_data(sales_data, test_size=0.2)

# Define a function to create a pipeline for the recommendation system
def create_pipeline():
    # Define the pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('lda', LatentDirichletAllocation()),
        ('classifier', RandomForestClassifier())
    ])
    return pipeline

# Create the pipeline
pipeline = create_pipeline()

# Define a function to train the pipeline
def train_pipeline(pipeline, train_data):
    # Train the pipeline
    pipeline.fit(train_data['text'], train_data['interaction_type'])
    return pipeline

# Train the pipeline
pipeline = train_pipeline(pipeline, train_user_interactions)

# Define a function to evaluate the pipeline
def evaluate_pipeline(pipeline, test_data):
    # Evaluate the pipeline
    predictions = pipeline.predict(test_data['text'])
    accuracy = accuracy_score(test_data['interaction_type'], predictions)
    precision = precision_score(test_data['interaction_type'], predictions)
    recall = recall_score(test_data['interaction_type'], predictions)
    f1 = f1_score(test_data['interaction_type'], predictions)
    return accuracy, precision, recall, f1

# Evaluate the pipeline
accuracy, precision, recall, f1 = evaluate_pipeline(pipeline, test_user_interactions)

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Plot the ROC curve
plt.figure(figsize=(10,6))
fpr, tpr, _ = roc_curve(test_user_interactions['interaction_type'], pipeline.predict_proba(test_user_interactions['text'])[:,1])
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot the precision-recall curve
plt.figure(figsize=(10,6))
precision, recall, _ = precision_recall_curve(test_user_interactions['interaction_type'], pipeline.predict_proba(test_user_interactions['text'])[:,1])
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()