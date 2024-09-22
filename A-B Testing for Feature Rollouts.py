# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Step 1: Simulate User Data
np.random.seed(42)

# Parameters for simulation
n_users = 10000  # Number of users in each group
control_conversion_rate = 0.30  # 30% of control group completes a purchase
test_conversion_rate = 0.35  # 35% of test group completes a purchase (5% increase in the new feature group)

# Simulate control group (existing checkout process)
control_group = pd.DataFrame({
    'group': 'control',
    'converted': np.random.binomial(1, control_conversion_rate, n_users)  # 1 if converted, 0 if not
})

# Simulate test group (new checkout process)
test_group = pd.DataFrame({
    'group': 'test',
    'converted': np.random.binomial(1, test_conversion_rate, n_users)  # 1 if converted, 0 if not
})

# Combine the two groups into one dataset
ab_test_data = pd.concat([control_group, test_group])

# Step 2: Calculate Conversion Rates
conversion_rates = ab_test_data.groupby('group')['converted'].mean()
print("Conversion Rates:")
print(conversion_rates)

# Step 3: Hypothesis Testing (Z-Test)
# Calculate the z-score and p-value to check statistical significance
control_success = control_group['converted'].sum()
test_success = test_group['converted'].sum()

control_size = len(control_group)
test_size = len(test_group)

# Pooled probability
p_pool = (control_success + test_success) / (control_size + test_size)

# Standard error
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/control_size + 1/test_size))

# Z-score
z_score = (conversion_rates['test'] - conversion_rates['control']) / se_pool

# P-value (two-tailed test)
p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

print(f"\nZ-Score: {z_score:.2f}")
print(f"P-Value: {p_value:.5f}")

# Step 4: Lift Calculation
lift = ((conversion_rates['test'] - conversion_rates['control']) / conversion_rates['control']) * 100
print(f"\nLift: {lift:.2f}%")

# Step 5: Visualize the Results
plt.figure(figsize=(8, 5))
sns.barplot(x=conversion_rates.index, y=conversion_rates.values)
plt.title('Conversion Rate by Group')
plt.xlabel('Group')
plt.ylabel('Conversion Rate')
plt.ylim(0, 0.5)
plt.show()
