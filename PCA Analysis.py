from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2023)

X = pd.read_csv('New Correlation3.csv')  # Replace 'YourDataset.csv' with your actual dataset file

# Transpose the dataset to have variables in columns
X = X.T

# Initialize PCA and fit the data
pca = PCA()
pca.fit(X)

# Access the explained variance ratio
explained_var_ratio = pca.explained_variance_ratio_

# Create scree plot
num_components = len(explained_var_ratio)
plt.plot(range(1, num_components + 1), explained_var_ratio, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')

# Add percentage labels
for i, ratio in enumerate(explained_var_ratio):
    plt.text(i + 1, ratio + 0.01, f'{ratio*100:.2f}%', ha='center')

plt.xticks(range(1, num_components + 1))
plt.show()

# Access the components matrix
components_matrix = pca.components_

# Create DataFrame with row and column labels
row_labels = [f"PC{i+1}" for i in range(components_matrix.shape[0])]
column_labels = [f"Variable{i+1}" for i in range(X.shape[1])]
components_df = pd.DataFrame(components_matrix, index=row_labels, columns=column_labels)

# Save components DataFrame to a CSV file
components_df.to_csv("components_matrix.csv")

print("Components matrix saved to 'components_matrix.csv'.")