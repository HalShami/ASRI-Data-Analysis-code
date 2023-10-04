import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
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


# Define the predictors (PC1 and PC2) and the target variable (Poverty)
X = data[['PC1', 'PC2']]
y = data['Poverty']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the neural network model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test, verbose=0)
r_squared = 1 - mse / np.var(y_test)
print("R-squared: ", r_squared)