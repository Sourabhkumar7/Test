import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Load the dataset
dataset = pd.read_csv('instagram_reach.csv')

# Split the data
X = dataset.drop(['likes', 'Time_Since_posted'], axis=1)
y_likes = dataset['likes']
y_time_since_posted = dataset['Time_Since_posted']

X_train, X_test, y_likes_train, y_likes_test, y_time_train, y_time_test = train_test_split(
    X, y_likes, y_time_since_posted, test_size=0.2, random_state=42
)

# Define preprocessing steps
numeric_features = ['Followers']
text_features = ['username', 'Caption', 'Hashtag']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['any_categorical_columns_here']
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder())
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models
likes_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

time_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train the models
likes_model.fit(X_train, y_likes_train)
time_model.fit(X_train, y_time_train)

# Make predictions
likes_predictions = likes_model.predict(X_test)
time_predictions = time_model.predict(X_test)

# Evaluate the models
mae_likes = mean_absolute_error(y_likes_test, likes_predictions)
rmse_likes = mean_squared_error(y_likes_test, likes_predictions, squared=False)

mae_time = mean_absolute_error(y_time_test, time_predictions)
rmse_time = mean_squared_error(y_time_test, time_predictions, squared=False)

print(f"Likes Model - MAE: {mae_likes}, RMSE: {rmse_likes}")
print(f"Time Model - MAE: {mae_time}, RMSE: {rmse_time}")

