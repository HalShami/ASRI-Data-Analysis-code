import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
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

# Create polynomial features
degree = 2  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Perform polynomial regression with cross-validation
k = 5  # Number of folds for cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Create a polynomial regression model
model = LinearRegression()

# Perform cross-validation
scores = cross_val_score(model, X_poly, Y, cv=kf, scoring='r2')

# Print the cross-validation scores
print("Cross-validation R-squared scores:", scores)
print("Mean R-squared score:", np.mean(scores))
