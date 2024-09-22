# Import necessary libraries
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (change the URL to a local file if necessary)
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
data = pd.read_excel(url)

# Data preparation
# Drop rows with missing CustomerID or InvoiceDate
data_clean = data.dropna(subset=['CustomerID', 'InvoiceDate'])

# Filter out negative quantity (to remove returns)
data_clean = data_clean[data_clean['Quantity'] > 0]

# Ensure CustomerID is an integer
data_clean['CustomerID'] = data_clean['CustomerID'].astype(int)

# Create a new column for Total Sales (Quantity * UnitPrice)
data_clean['TotalSales'] = data_clean['Quantity'] * data_clean['UnitPrice']

# RFM Calculation
# Set the reference date as one day after the last transaction
reference_date = data_clean['InvoiceDate'].max() + dt.timedelta(days=1)

# Group by CustomerID to calculate Recency, Frequency, and Monetary value
rfm_table = data_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency: days since last purchase
    'InvoiceNo': 'nunique',  # Frequency: number of unique invoices
    'TotalSales': 'sum'  # Monetary: total money spent
})

# Rename columns to 'Recency', 'Frequency', 'Monetary'
rfm_table.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalSales': 'Monetary'
}, inplace=True)

# RFM Scoring
# Create RFM score categories (1 to 5)
rfm_table['R_Quartile'] = pd.qcut(rfm_table['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_table['F_Quartile'] = pd.qcut(rfm_table['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_table['M_Quartile'] = pd.qcut(rfm_table['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine RFM quartiles into a single RFM score
rfm_table['RFM_Score'] = rfm_table['R_Quartile'].astype(str) + rfm_table['F_Quartile'].astype(str) + rfm_table['M_Quartile'].astype(str)

# Customer Segmentation
def rfm_segment(df):
    if df['RFM_Score'] == '555':
        return 'Best Customers'
    elif df['RFM_Score'].startswith('5'):
        return 'Loyal Customers'
    elif df['RFM_Score'].endswith('1'):
        return 'Low-Value Customers'
    elif df['R_Quartile'] == '1':
        return 'At Risk'
    else:
        return 'Others'

# Apply the segmentation function
rfm_table['Customer_Segment'] = rfm_table.apply(rfm_segment, axis=1)

# Display the RFM table with segments
print(rfm_table.head())

# Visualization of Customer Segments
plt.figure(figsize=(10, 6))
sns.countplot(data=rfm_table, x='Customer_Segment', order=rfm_table['Customer_Segment'].value_counts().index)
plt.title('Customer Segments Distribution')
plt.xlabel('Customer Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()
