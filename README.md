# House Price Prediction using Linear Regression

## Project Overview

This project implements a machine learning model to predict house prices based on key features: **Ground Living Area (square footage)**, **Number of Bedrooms**, and **Number of Full Bathrooms**. It utilizes a simple yet powerful **Linear Regression** algorithm, a fundamental model in machine learning, to demonstrate an end-to-end predictive analytics pipeline.

The project is structured into two main parts:
1.  **`main.py`**: Handles data loading, exploratory data analysis (EDA), data preprocessing (including outlier removal), model training, evaluation, and finally, saves the trained model to a file.
2.  **`predict.py`**: A user-friendly script that loads the pre-trained model and allows users to interactively input house features to get an immediate price prediction.

## Features Used

* **`GrLivArea`**: Above grade (ground) living area in square feet.
* **`BedroomAbvGr`**: Number of bedrooms above grade (does not include basement bedrooms).
* **`FullBath`**: Number of full bathrooms above grade.
* **Target Variable**: `SalePrice` (the price of the house).

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

You need Python 3 installed on your system.

### 1. Clone the Repository (or create project folder)

If this were a Git repository, you would clone it. For your current setup, ensure you have a project folder named `House Price Prediction using Linear Regression`.

### 2. Download the Dataset

The project uses the "House Prices - Advanced Regression Techniques" dataset from Kaggle.

* Go to: [https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
* **Log in or register** on Kaggle.
* Download the file named **`train.csv`**.
* Place the `train.csv` file directly into your `House Price Prediction using Linear Regression` project folder.

### 3. Install Dependencies

It's highly recommended to use a virtual environment to manage project dependencies.

* **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    ```
* **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
* **Install required packages:**
    The `requirements.txt` file lists all necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```
    This command will install all the necessary libraries like `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.

## How to Run the Project

### Step 1: Train the Model

First, you need to train the machine learning model. This process involves loading the data, preprocessing it, training the Linear Regression model, evaluating its performance, and saving the trained model to a file (`house_price_model.pkl`).

* Open your terminal or command prompt.
* Navigate to your project directory:
    ```bash
    cd "House Price Prediction using Linear Regression"
    ```
* Run the main training script:
    ```bash
    python main.py
    ```
    This script will print various outputs related to data loading, exploration, model evaluation (MSE, RMSE, R-squared), and display several plots (scatter plots and actual vs. predicted plot). It will also confirm when the model is saved.

### Step 2: Make Predictions

Once the model is trained and saved, you can use the `predict.py` script to get interactive house price predictions.

* Ensure you are still in your project directory in the terminal.
* Run the prediction script:
    ```bash
    python predict.py
    ```
* The script will prompt you to enter the `Ground Living Area`, `Number of Bedrooms`, and `Number of Full Bathrooms`. Provide these values, and the script will output the estimated `Sale Price`.

    **Example Interaction:**
    ```
    --- House Price Prediction Tool ---
    Model 'house_price_model.pkl' loaded successfully.

    Please enter the details for the house:
    Enter Ground Living Area (sq ft, e.g., 2000): 2200
    Enter Number of Bedrooms (e.g., 3): 4
    Enter Number of Full Bathrooms (e.g., 2.0 or 1.5): 2.5

    --- Prediction Result ---
    Based on your input:
      - Ground Living Area: 2200.0 sq ft
      - Bedrooms: 4
      - Full Bathrooms: 2.5
    Estimated Sale Price: $300,000.00  # (Actual predicted value will vary)

    Thank you for using the House Price Prediction Tool!
    ```

Make sure you've run pip freeze > requirements.txt so that requirements.txt actually exists.
