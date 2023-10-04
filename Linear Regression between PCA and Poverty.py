import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

# Read data from CSV file
data = pd.read_csv('New Correlation.csv')

# Select the variables for correlation with principal components
variables = ['BlackPct', 'AsianPct', 'HispanicPct', 'NativeAmericanPct', 'MultipleRacePct', 'MinorityPct',
             'WhitePct', 'SimpDivInd2', 'ForeignBornPct', 'College+Pct', 'Age 65+ Pct']

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data[variables])

# Extract PC1 and PC2 from the principal components
PC1 = principal_components[:, 0]
PC2 = principal_components[:, 1]

# Add PC1 and PC2 as columns to the dataset
data['PC1'] = PC1
data['PC2'] = PC2

# Define the independent variables (X) and the dependent variable (Y)
X = data[['PC1', 'PC2']]
Y = data['Poverty']

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Create and fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the model summary
print(model.summary())