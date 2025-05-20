# main.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("--- Starting House Price Prediction Project ---")

# Step 1: Data Collection & Loading
# Make sure 'train.csv' is in the same directory as this script (main.py)
try:
    df = pd.read_csv('train.csv')
    print("\nDataset 'train.csv' loaded successfully!")
    print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
except FileNotFoundError:
    print("\nError: 'train.csv' not found. Please make sure the file is in the same directory as 'main.py'.")
    print("You can download it from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data (look for train.csv)")
    exit() # Exit the script if the file isn't found


# Step 2: Initial Data Exploration

print("\n--- First 5 rows of the dataset (df.head()) ---")
print(df.head())

print("\n--- Basic information about the dataset (df.info()) ---")
df.info()

print("\n--- Descriptive statistics of numerical columns (df.describe()) ---")
print(df.describe())


# Step 3: Focused Data Exploration and Feature Selection

print("\n--- Checking for missing values in selected columns ---")
selected_features_and_target = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']
print(df[selected_features_and_target].isnull().sum())
# For these specific columns, you should find 0 missing values, which is ideal!

# Visualize relationships with SalePrice
print("\n--- Generating scatter plots for feature relationships ---")

# GrLivArea vs SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('SalePrice vs. GrLivArea')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show() # This command displays the plot

# BedroomAbvGr vs SalePrice (using boxplot for discrete categories)
plt.figure(figsize=(10, 6))
sns.boxplot(x='BedroomAbvGr', y='SalePrice', data=df)
plt.title('SalePrice vs. Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()

# FullBath vs SalePrice (using boxplot for discrete categories)
plt.figure(figsize=(10, 6))
sns.boxplot(x='FullBath', y='SalePrice', data=df)
plt.title('SalePrice vs. Number of Full Bathrooms')
plt.xlabel('Number of Full Bathrooms')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()

# Outlier Removal for GrLivArea (common practice for this Kaggle dataset)
initial_rows = df.shape[0]
df = df[df['GrLivArea'] < 4000] # Remove houses with extremely large living areas that might be outliers
print(f"\nRemoved {initial_rows - df.shape[0]} rows considered outliers in GrLivArea (GrLivArea >= 4000).")
print(f"New number of rows: {df.shape[0]}")


# Step 4: Data Preparation for Modeling

# Define our features (X) and target (y)
# X will contain our independent variables (what we use to predict)
# y will contain our dependent variable (what we want to predict)
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing, 80% for training
# random_state=42 ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size (X_train, y_train): {X_train.shape[0]} rows")
print(f"Testing set size (X_test, y_test): {X_test.shape[0]} rows")
print(f"Features used for training: {X_train.columns.tolist()}")



# Step 5: Train the Linear Regression Model

print("\n--- Training the Linear Regression Model ---")

# Create a Linear Regression model object
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print("Model training complete!")


# Step 6: Evaluate the Model

print("\n--- Evaluating the Model ---")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # RMSE is often more interpretable as it's in the same units as the target
r2 = r2_score(y_test, y_pred) # R-squared tells us how much variance our model explains

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Display the model's coefficients and intercept
print("\n--- Model Coefficients and Intercept ---")
print("These show the estimated impact of each feature on SalePrice.")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")


# Step 7: Visualize Predictions

print("\n--- Visualizing Actual vs. Predicted Prices ---")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7) # Plot actual vs predicted values
# Plot a red dashed line representing the ideal scenario (actual == predicted)


# Step 8: Make a Simple Prediction and Conclusion

print("\n--- Making a Sample Prediction ---")

# Let's predict the price of a new house:
# Example house: 2000 sq ft, 3 bedrooms, 2 full bathrooms
new_house_features = pd.DataFrame([[2000, 3, 2]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predicted_price = model.predict(new_house_features)

print(f"Predicted price for a house with:")
print(f"  - GrLivArea: {new_house_features['GrLivArea'].iloc[0]} sq ft")
print(f"  - Bedrooms: {new_house_features['BedroomAbvGr'].iloc[0]}")
print(f"  - Bathrooms: {new_house_features['FullBath'].iloc[0]}")
print(f"Predicted Sale Price: ${predicted_price[0]:,.2f}") # Format as currency

print("\n--- Project Complete ---")
print("You have successfully implemented a Linear Regression model to predict house prices!")
print("Next steps could involve:")
print("  - Adding more features (e.g., year built, neighborhood, lot area).")
print("  - Handling categorical features (e.g., using one-hot encoding).")
print("  - Trying other regression models (e.g., Random Forest, Gradient Boosting).")
print("  - Performing more advanced feature engineering.")
print("  - Cross-validation for more robust evaluation.")
model_filename = 'house_price_model.pkl' # .pkl is a common extension for pickled objects
joblib.dump(model, model_filename)

print(f"\nModel successfully saved as '{model_filename}'")