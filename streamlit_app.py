import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Sample DataFrame creation (replace this with your actual data loading method)
# Let's assume 'df' is your DataFrame with features and target columns
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.rand(100)
})

# Define feature columns and target column
feature_columns = ['feature1', 'feature2']
target_column = 'target'

# 1. Drop rows with missing values in both features and target (ensure no NaNs)
df = df.dropna(subset=feature_columns + [target_column])

# 2. Extract features and target
features = df[feature_columns].to_numpy()  # Convert features to NumPy array
target = df[target_column].to_numpy()  # Convert target to NumPy array

# 3. Ensure the alignment of features and target (same number of samples)
assert features.shape[0] == target.shape[0], f"Mismatch in the number of samples: {features.shape[0]} features vs {target.shape[0]} target values"

# 4. Split data into training and testing sets (80% train, 20% test, no shuffling)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# 5. Normalize the feature data (if needed, for example using StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Initialize and train a model (example: Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict on test data
predictions = model.predict(X_test)

# 8. Output model performance (for example, print R^2 score)
r2_score = model.score(X_test, y_test)
print(f"R^2 Score on Test Data: {r2_score:.4f}")

# 9. Optionally, return the data and predictions for further use
def get_data(selected_symbol, start_date):
    # You can replace this with your actual data retrieval logic
    # In this example, we just return the data and predictions.
    return df, predictions

# Example usage
selected_symbol = 'AAPL'  # Replace with actual selected symbol
start_date = '2020-01-01'  # Replace with actual start date
data, predictions = get_data(selected_symbol, start_date)

# Output the data and predictions
print("Data and Predictions:\n", data.head())
