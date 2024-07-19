import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np


columns = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
       'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
       'inq.last.6mths', 'delinq.2yrs', 'pub.rec']


st.write("""
         ### Has the loan been paid or not? 💸🧾
         This app predicts the loan payment""")

st.sidebar.header("User Input Features")

st.sidebar.markdown(""" [Example CSV input file](https://github.com/morteza-rp/fanap/blob/main/DATASCIENCE2_DP_Ex02/loan_example.csv)""")

# Collect user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        
       credit_policy = st.sidebar.select_slider("Select credit.policy:", [0, 1], 
                                          help="a binary indicator (0 or 1) indicating whether the individual meets the credit underwriting criteria of the lending institution (1) or not (0).")

       purpose = st.sidebar.selectbox("Select purpose:", ["debt_consolidation",
                                   "all_other",         
                                   "credit_card",       
                                   "home_improvement", 
                                   "small_business",   
                                   "major_purchase",   
                                   "educational" ], 
                                          help= "the purpose for which the loan was taken. Common purposes might include debt consolidation, home improvement, education, etc.")

       int_rate = st.sidebar.number_input("Input int.rate:", 
                                          help="the interest rate on the loan, expressed as a percentage. It indicates the cost of borrowing for the individual.")

       installment = st.sidebar.number_input("Input installment:", 
                                          help= "the monthly installment amount that the individual has to pay towards repaying the loan.")

       log_annual_inc = st.sidebar.number_input("Input log.annual.inc:", 
                                          help= "the natural logarithm of the individual's annual income.")

       dti = st.sidebar.number_input("Input dti:", 
                                          help= "the debt-to-income ratio")

       fico = st.sidebar.number_input("Input fico:", 
                                          help= "the FICO credit score of the individual. FICO scores are commonly used by lenders to assess credit risk and determine loan eligibility and interest rates.")

       days_with_cr_line = st.sidebar.number_input("Input days.with.cr.line:", 
                                          help= "the number of days the individual has had a credit line.")

       revol_bal = st.sidebar.number_input("Input revol.bal:", 
                                          help= "the revolving balance, which is the outstanding balance on the individual's revolving credit accounts (e.g., credit cards) at the time of data collection.")

       revol_util = st.sidebar.number_input("Input revol.util:", 
                                          help= "the revolving utilization rate, which is the ratio of the individual's revolving credit balances to their credit limits. It indicates how much of their available credit they are currently using.")

       inq_last_6mths = st.sidebar.number_input("Input inq.last.6mths:", 
                                          help="the number of inquiries made by creditors in the last 6 months.")

       delinq_2yrs = st.sidebar.number_input("Input delinq.2yrs:", 
                                          help="the number of times the individual has been delinquent on payments in the past 2 years.")

       pub_rec = st.sidebar.number_input("Input pub.rec", 
                                          help="the number of derogatory public records (e.g., bankruptcies, tax liens) associated with the individual's credit report.")

       features = pd.DataFrame([[credit_policy,purpose,int_rate,installment,log_annual_inc,dti,fico,days_with_cr_line,revol_bal,revol_util,inq_last_6mths,delinq_2yrs,pub_rec]], 
                                columns=columns)

       return features

    input_df = user_input_features()


# combines user input features with entire loan dataset
loan_raw = pd.read_csv(r"E:\github\fanap\DATASCIENCE2_DP_Ex02\loan_data.csv")
loan = loan_raw.drop(columns=["not.fully.paid"])
df = pd.concat([input_df, loan], axis=0)

# Encoding of ordinal features
le = LabelEncoder()
df['purpose'] = le.fit_transform(df['purpose'])
df = df[:1] # selects only the first row(the user input data)

# Display user input features
st.subheader("User input features")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)")
    st.write(df)

# reads in saved classification model
xgb_clf = XGBClassifier()
xgb_clf.load_model(r"E:\github\fanap\DATASCIENCE2_DP_Ex02\xgb_model.json")

# Apply model to make prediction
prediction = xgb_clf.predict(df)
prediction_proba = xgb_clf.predict_proba(df)


st.subheader("Prediction")
not_fully_paid = np.array(["NO", "YES"])
st.write(not_fully_paid[prediction])

st.subheader("Prediction Probability")
st.write(prediction_proba)