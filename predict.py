# predict.py

import pandas as pd
import joblib
import sys # Used for sys.exit()

print("--- House Price Prediction Tool ---")

# Load the trained model
model_filename = 'house_price_model.pkl'
try:
    model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.")
    print("Please ensure you have run 'main.py' at least once to train and save the model.")
    sys.exit(1) # Exit the script if the model file is not found

# Get user input for features
print("\nPlease enter the details for the house:")
try:
    gr_liv_area = float(input("Enter Ground Living Area (sq ft, e.g., 2000): "))
    bedrooms = int(input("Enter Number of Bedrooms (e.g., 3): "))
    bathrooms = float(input("Enter Number of Full Bathrooms (e.g., 2.0 or 1.5): "))
except ValueError:
    print("Invalid input. Please enter numerical values for area, bedrooms, and bathrooms.")
    sys.exit(1) # Exit if input is not valid numbers

# Create a DataFrame for the user's input, matching the training features
# Column names MUST exactly match the features used in training
user_input_df = pd.DataFrame([[gr_liv_area, bedrooms, bathrooms]],
                             columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])

# Make the prediction
predicted_price = model.predict(user_input_df)

# Display the prediction
print("\n--- Prediction Result ---")
print(f"Based on your input:")
print(f"  - Ground Living Area: {gr_liv_area} sq ft")
print(f"  - Bedrooms: {bedrooms}")
print(f"  - Full Bathrooms: {bathrooms}")
print(f"Estimated Sale Price: ${predicted_price[0]:,.2f}")
print("\nThank you for using the House Price Prediction Tool!")