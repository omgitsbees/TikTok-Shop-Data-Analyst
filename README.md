Customer Segmentation Using RFM Analysis

This project demonstrates how to perform Customer Segmentation using RFM Analysis (Recency, Frequency, Monetary) on an e-commerce dataset. RFM analysis is a data-driven technique to categorize customers based on their purchasing behavior, allowing businesses to tailor marketing strategies for different customer segments.
Table of Contents

    Overview
    Dataset
    RFM Methodology
    Project Structure
    Installation
    Usage
    Results
    Contributing
    License

Overview

RFM analysis is a marketing technique used to quantify customer engagement by segmenting them into different groups. This project analyzes the purchasing patterns of customers to segment them based on:

    Recency: How recently the customer made a purchase.
    Frequency: How often the customer made purchases.
    Monetary: How much the customer has spent on purchases.

Using these metrics, customers can be categorized into various segments such as Best Customers, At Risk, Loyal Customers, and more.
Key Steps:

    Data Preparation: Clean and preprocess the data.
    RFM Calculation: Calculate Recency, Frequency, and Monetary value for each customer.
    RFM Scoring: Assign scores based on quantiles for each metric.
    Customer Segmentation: Classify customers based on their RFM score.
    Visualization: Display customer segments using visual plots.

Dataset

The dataset used is the Online Retail Dataset from the UCI Machine Learning Repository. It contains transactional data from a UK-based online retailer over the course of one year (2010-2011).
Dataset Features:

    InvoiceNo: Invoice number (unique identifier for each transaction)
    StockCode: Product (item) code
    Description: Product (item) description
    Quantity: Quantity of each product purchased
    InvoiceDate: The date the invoice was generated
    UnitPrice: Price per unit of the product
    CustomerID: Unique identifier for the customer
    Country: Country of residence of the customer

RFM Methodology

The steps to calculate RFM metrics are:

    Recency: Days since the last purchase.
    Frequency: Total number of purchases.
    Monetary: Total spending by the customer.

Each customer is scored for Recency, Frequency, and Monetary values, and based on their RFM score, they are segmented into different groups.
Project Structure

bash

├── data/
│   └── Online Retail Dataset (or link to data source)
├── rfm_analysis.py  # Main script for RFM analysis and customer segmentation
├── README.md        # Project readme file
└── requirements.txt # Python dependencies

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/customer-segmentation-rfm.git
cd customer-segmentation-rfm

Create a virtual environment (optional but recommended):

bash

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install the required dependencies:

bash

    pip install -r requirements.txt

    Download the dataset:
        The dataset can be downloaded from the UCI Online Retail Dataset. Place it in the data/ folder.

Usage

To run the RFM analysis and segment customers:

    Ensure the dataset is in the correct folder (data/Online Retail Dataset.xlsx).

    Run the Python script:

    bash

    python rfm_analysis.py

This will clean the dataset, calculate the RFM metrics, assign scores, and classify customers into segments. Additionally, it will generate a plot displaying the distribution of customer segments.
Output:

    RFM table with scores and customer segments.
    A bar plot visualizing the distribution of customers across segments.

Results

Once the analysis is complete, customers will be segmented into categories such as:

    Best Customers: Customers with high Recency, Frequency, and Monetary values.
    Loyal Customers: Customers who purchase frequently.
    At Risk: Customers who haven't purchased recently.
    Low-Value Customers: Customers who purchase infrequently and spend less.

Sample Visual Output:

![Screenshot 2024-09-22 150759](https://github.com/user-attachments/assets/eeb3c582-3a2a-4bbe-ad57-1d4cb7d29e29)

------------------------------------------------------------------------------------------------------------------

Cohort Analysis for Customer Retention

This project demonstrates how to perform Cohort Analysis to track customer retention over time for an e-commerce platform. Cohort analysis groups customers based on their first purchase date and helps identify how different groups (cohorts) behave across subsequent time periods. This type of analysis is crucial for understanding customer retention patterns and improving long-term customer engagement.
Table of Contents

    Overview
    Dataset
    Cohort Analysis Methodology
    Project Structure
    Installation
    Usage
    Results
    Contributing
    License

Overview

Cohort analysis is a technique used to understand customer behavior by grouping them into cohorts, usually based on their first purchase date. By tracking these cohorts over time, businesses can gain insights into customer retention, loyalty, and churn.

This project uses the Online Retail Dataset to group customers into cohorts based on the month of their first purchase. We then analyze how long customers from each cohort continue to make purchases in subsequent months.
Key Steps:

    Data Preparation: Clean the data and format it for cohort analysis.
    Cohort Definition: Define cohorts based on the month of the customer's first purchase.
    Retention Calculation: Calculate the retention rate of each cohort over time.
    Visualization: Create a heatmap to visualize the retention rates for each cohort over several months.

Dataset

The dataset used is the Online Retail Dataset from the UCI Machine Learning Repository. It contains transactional data from a UK-based online retailer over the course of one year (2010-2011).
Dataset Features:

    InvoiceNo: Invoice number (unique identifier for each transaction)
    StockCode: Product code
    Description: Product description
    Quantity: Quantity of the product purchased
    InvoiceDate: Date the invoice was generated
    UnitPrice: Price per unit of the product
    CustomerID: Unique identifier for the customer
    Country: Country of the customer

Cohort Analysis Methodology

    Cohort Definition:
        Customers are grouped into cohorts based on their first purchase month.
        We calculate the CohortIndex, which represents how many months have passed since the customer’s first purchase.

    Retention Calculation:
        For each cohort, the number of returning customers is tracked for each subsequent month.
        The retention rate is calculated as the percentage of customers who make repeat purchases in subsequent months.

    Visualization:
        The retention rates for each cohort are displayed in a heatmap, where each row represents a cohort and each column represents months since the first purchase.

Project Structure

bash

├── data/
│   └── Online Retail Dataset (or link to data source)
├── cohort_analysis.py  # Main script for cohort analysis
├── README.md           # Project readme file
└── requirements.txt    # Python dependencies

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/cohort-analysis-retention.git
cd cohort-analysis-retention

Create a virtual environment (optional but recommended):

bash

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install the required dependencies:

bash

    pip install -r requirements.txt

    Download the dataset:
        The dataset can be downloaded from the UCI Online Retail Dataset. Place it in the data/ folder.

Usage

To run the Cohort Analysis and visualize the customer retention rates:

    Ensure the dataset is in the correct folder (data/Online Retail Dataset.xlsx).

    Run the Python script:

    bash

    python cohort_analysis.py

This script will:

    Clean the data.
    Calculate customer cohorts and retention rates.
    Display a heatmap of retention rates over time.

Output:

    Cohort Table: A table showing the number of customers retained over time.
    Retention Heatmap: A heatmap that visualizes the retention rates of each cohort.

Sample Visualization:

The heatmap will look similar to the example below:

Results

The output of this project is a heatmap that shows the retention rates for different customer cohorts over time. You can use the heatmap to understand:

    Which cohorts have the highest retention rates.
    How customer retention changes over time.
    At which point customers are most likely to stop purchasing (churn).

This insight can help in designing better customer retention strategies, such as re-engagement campaigns or personalized offers to at-risk customers.

![Screenshot 2024-09-22 151519](https://github.com/user-attachments/assets/690616f7-788d-46e3-838e-31a64865f33d)
