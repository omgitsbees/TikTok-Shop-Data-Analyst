# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (Online Retail dataset from UCI)
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
data = pd.read_excel(url)

# Data Preparation
# Drop rows with missing CustomerID or InvoiceDate
data_clean = data.dropna(subset=['CustomerID', 'InvoiceDate'])

# Ensure CustomerID is an integer
data_clean['CustomerID'] = data_clean['CustomerID'].astype(int)

# Filter positive sales only (Quantity > 0)
data_clean = data_clean[data_clean['Quantity'] > 0]

# Add a new column for Total Sales (Quantity * UnitPrice)
data_clean['TotalSales'] = data_clean['Quantity'] * data_clean['UnitPrice']

# Convert InvoiceDate to datetime if it's not already
data_clean['InvoiceDate'] = pd.to_datetime(data_clean['InvoiceDate'])

# Step 2: Define Cohorts
# Create a new column for the first purchase month (cohort month) for each customer
data_clean['OrderMonth'] = data_clean['InvoiceDate'].dt.to_period('M')

# Create a column for CohortMonth which is the month of the customer's first purchase
data_clean['CohortMonth'] = data_clean.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')

# Step 3: Calculate the Cohort Index
# The CohortIndex represents the number of months since the customer's first purchase (0 = the month of the first purchase)
def get_date_int(df, column):
    """Helper function to extract year, month, and day as integers."""
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month

# Extract the year and month for CohortMonth and OrderMonth
cohort_year, cohort_month = get_date_int(data_clean, 'CohortMonth')
order_year, order_month = get_date_int(data_clean, 'OrderMonth')

# Calculate the difference in months between the CohortMonth and the OrderMonth
data_clean['CohortIndex'] = (order_year - cohort_year) * 12 + (order_month - cohort_month)

# Step 4: Create a Cohort Table
# Group by CohortMonth and CohortIndex to count the number of unique customers for each cohort in each month
cohort_data = data_clean.groupby(['CohortMonth', 'CohortIndex']).agg({
    'CustomerID': pd.Series.nunique
}).reset_index()

# Create a pivot table to display the cohort table
cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='CustomerID')

# Step 5: Calculate Retention Rates
# The retention rate is the number of customers in a cohort who make a repeat purchase divided by the number of customers in the cohort's first month
cohort_size = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_size, axis=0)

# Step 6: Visualize the Retention Matrix
plt.figure(figsize=(12, 8))
plt.title('Cohort Analysis - Retention Rates')
sns.heatmap(retention, annot=True, fmt='.0%', cmap='Blues')
plt.xlabel('Cohort Index (Months since First Purchase)')
plt.ylabel('Cohort Month')
plt.show()
