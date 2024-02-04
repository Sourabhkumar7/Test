import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
housing_data = pd.read_csv('Bengaluru_Housing_Dataset.csv')

# Exploratory Data Analysis (EDA)

# Overview of the dataset
print(housing_data.head())

# Summary statistics
print(housing_data.describe())

# Check for missing values
print(housing_data.isnull().sum())

# Visualize the distribution of the target variable (price)
sns.histplot(housing_data['price'], kde=True)
plt.title('Distribution of Housing Prices')
plt.show()

# Explore the relationship between features and the target variable
plt.figure(figsize=(12, 6))
sns.scatterplot(x='total_sqft', y='price', data=housing_data)
plt.title('Total Sqft vs Price')
plt.show()

# Feature Engineering

# Handle missing values
housing_data = housing_data.dropna()

# Extract meaningful features from 'total_sqft'
housing_data['total_sqft'] = housing_data['total_sqft'].apply(lambda x: eval(x.replace('Sq. Meter', '').replace('Sq. Yards', '').replace('Perch', '').replace('Acres', '').replace('Cents', '').replace('Guntha', '').replace('Grounds', '').replace('sqft', '').replace('BHK', '').replace(' Bedroom', '').replace(',', '')))

# Feature scaling
scaler = MinMaxScaler()
numeric_features = ['total_sqft', 'bath', 'balcony', 'price']
numeric_transformer = Pipeline(steps=[('scaler', scaler)])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ])

# Convert categorical features to numerical using one-hot encoding
categorical_features = ['location', 'availability', 'size', 'society']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

# Split the data into features (X) and target variable (y)
X = housing_data.drop(['price'], axis=1)
y = housing_data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM regressor
svm_regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='linear'))
])

svm_regressor.fit(X_train, y_train)

# Make predictions
predictions = svm_regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
