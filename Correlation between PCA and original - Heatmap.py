import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('New Correlation3.csv')

# Select the variables for correlation with principal components
variables = ['BlackPct', 'AsianPct', 'HispanicPct', 'NativeAmericanPct', 'MultipleRacePct', 'MinorityPct',
             'WhitePct', 'SimpDivInd2', 'ForeignBornPct', 'College+Pct', 'Age 65+ Pct']

# Compute correlation matrix
correlation_matrix = data[variables].corr()

# Perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data[variables])

# Compute the correlation between attributes and principal components
correlation_pc = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=variables)

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_pc, annot=True, cmap='coolwarm', fmt='.3f', cbar=True, square=True)
plt.title('Correlation between Attributes and Principal Components')
plt.show()