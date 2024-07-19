Here's a draft README for your GitHub repository:

# Loan Payment Prediction App

This Streamlit app predicts whether a loan will be fully paid or not based on various input features.

## Overview

The app uses an XGBoost classifier model to predict loan payment status. Users can input loan details either through a CSV file upload or by manually entering values for each feature.

## Features

- Predict loan payment status (fully paid or not)
- Accept user input via CSV file or manual entry
- Display prediction and prediction probability
- Interactive UI with Streamlit

## Installation

To run this app locally, follow these steps:

1. Clone this repository
2. Install the required packages:
   ```
   pip install streamlit pandas scikit-learn xgboost numpy
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Upload a CSV file with loan data or manually input loan details using the sidebar.
2. The app will display the input features and the prediction.
3. The prediction indicates whether the loan is expected to be fully paid or not.
4. The app also shows the prediction probability.

## Input Features

The model uses the following features for prediction:

- credit.policy
- purpose
- int.rate
- installment
- log.annual.inc
- dti
- fico
- days.with.cr.line
- revol.bal
- revol.util
- inq.last.6mths
- delinq.2yrs
- pub.rec

## Model

The app uses an XGBoost classifier model trained on historical loan data. The model is loaded from a saved JSON file.

## Files

- `app.py`: The main Streamlit application
- `loan_data.csv`: Dataset used for combining with user input
- `xgb_model.json`: Saved XGBoost model
- `loan_example.csv`: Example CSV file for user input

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/morteza-rp/loan-payment/issues) if you want to contribute.
