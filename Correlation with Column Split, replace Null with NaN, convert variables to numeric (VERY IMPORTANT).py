import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv("HealthData_coded.csv", encoding='latin-1')

# Replace '#NULL!' with NaN
data = data.replace('#NULL!', np.nan)

# Convert "Age" column into separate columns
data = pd.get_dummies(data, columns=['Age'], prefix='Age')
data = pd.get_dummies(data, columns=['White'], prefix='White')

# Clean column names by removing leading/trailing whitespace
data.columns = data.columns.str.strip()

# Update the variables list with the appropriate column names
variables = ['Age_1', 'Age_2','Age_3','Age_4','Age_5','Age_6','White_0', 'White_1', 'GMan', 'GWoman', 'C19Vaccine']

# Convert variables to numeric if necessary
data[variables] = data[variables].apply(pd.to_numeric, errors='coerce')

# Compute correlation matrix
correlation_matrix = data[variables].corr()

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f', cbar=True, square=True)
plt.title('Correlation Heatmap')
plt.show()