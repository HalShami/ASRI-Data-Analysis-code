import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Correlation - Top 500.csv")

# Exclude non-numeric columns from correlation calculation
numeric_columns = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2%')
plt.xlabel(' ')
plt.ylabel(' ')
plt.title('Correlation Heatmap')
plt.show()

print(corr_matrix)