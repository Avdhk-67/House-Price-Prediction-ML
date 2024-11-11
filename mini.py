import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load training and test data
train_data = pd.read_csv('train.csv')  # Training data file
test_data = pd.read_csv('test.csv')    # Test data file

# Separate features (X) and target (y) for training data
X_train = train_data[['num_rooms', 'sqft', 'neighborhood', 'other_features', 'proximity_to_city']]
y_train = train_data['price']

# Separate features (X) and target (y) for test data
X_test = test_data[['num_rooms', 'sqft', 'neighborhood', 'other_features', 'proximity_to_city']]
y_test = test_data['price']

# Preprocessing pipeline for categorical features
preprocessor = ColumnTransformer(
    transformers=[ 
        ('cat', OneHotEncoder(drop='first'), ['neighborhood', 'other_features']) 
    ],
    remainder='passthrough'  # Leave other features (like num_rooms, sqft, proximity_to_city) as they are
)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Dictionary to store results
results = {}

# Linear Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
mse, r2 = evaluate_model(lr_pipeline, X_train, X_test, y_train, y_test)
results['Linear Regression'] = {'MSE': mse, 'R2': r2}

# Decision Tree
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])
mse, r2 = evaluate_model(dt_pipeline, X_train, X_test, y_train, y_test)
results['Decision Tree'] = {'MSE': mse, 'R2': r2}

# Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])
mse, r2 = evaluate_model(rf_pipeline, X_train, X_test, y_train, y_test)
results['Random Forest'] = {'MSE': mse, 'R2': r2}

# Display results
for model_name, metrics in results.items():
    print(f"{model_name} - MSE: {metrics['MSE']:.2f}, R2: {metrics['R2']:.2f}")



# predictioin
new_house_data = pd.DataFrame({
    'num_rooms': [5],             # Number of rooms
    'sqft': [3375],               # Square footage
    'neighborhood': ['Rural'],    # Neighborhood type (Urban or Rural)
    'other_features': ['No Garage'],  # Garage feature (Has Garage or No Garage)
    'proximity_to_city': [31]     # Proximity to city (numeric value)
})
 
predicted_price = rf_pipeline.predict(new_house_data)

# Output the prediction
print("")
print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")