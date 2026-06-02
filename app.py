import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 House Price Prediction System")
st.write("Predict house sale price using Machine Learning")

# Sidebar
st.sidebar.title("About Project")
st.sidebar.info("""
This project predicts house prices using a trained Machine Learning model.

Developer: Uchit Vyas  
Tool: Streamlit  
Model: House Price Prediction Model
""")

# Load saved files
model = joblib.load("House_Price_Prediction_Model.pkl")
scaler = joblib.load("House_Price_Prediction_Scaler.pkl")
columns = joblib.load("House_Price_Prediction_Columns.pkl")

# Load training data
train_df = pd.read_csv("train.csv")

# Input section
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider("Overall Quality", 1, 10, 5)
    GrLivArea = st.number_input("Living Area", value=1500)
    GarageCars = st.slider("Garage Cars", 0, 5, 2)
    TotalBsmtSF = st.number_input("Basement Area", value=800)
    YearBuilt = st.number_input("Year Built", value=2000)

with col2:
    FullBath = st.slider("Full Bathrooms", 0, 4, 2)
    YearRemodAdd = st.number_input("Year Remodeled", value=2005)
    GarageArea = st.number_input("Garage Area", value=400)
    FirstFlrSF = st.number_input("First Floor Area", value=1000)
    TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 1, 15, 6)

# Prepare input data
input_data = train_df.drop("SalePrice", axis=1).copy()

# Fill missing values safely
for col in input_data.columns:
    if input_data[col].dtype == "object":
        input_data[col] = input_data[col].fillna(input_data[col].mode()[0])
    else:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce")
        input_data[col] = input_data[col].fillna(input_data[col].median())
        
# Take one default row
single_row = input_data.iloc[[0]].copy()

# Replace values with user input
single_row["OverallQual"] = OverallQual
single_row["GrLivArea"] = GrLivArea
single_row["GarageCars"] = GarageCars
single_row["TotalBsmtSF"] = TotalBsmtSF
single_row["FullBath"] = FullBath
single_row["YearBuilt"] = YearBuilt
single_row["YearRemodAdd"] = YearRemodAdd
single_row["GarageArea"] = GarageArea
single_row["1stFlrSF"] = FirstFlrSF
single_row["TotRmsAbvGrd"] = TotRmsAbvGrd

# Convert categorical data
single_row_encoded = pd.get_dummies(single_row)

# Match training columns
single_row_encoded = single_row_encoded.reindex(columns=columns, fill_value=0)

# Scale data
single_row_scaled = scaler.transform(single_row_encoded)

# Prediction button
if st.button("Predict House Price"):
    prediction = model.predict(single_row_scaled)

    st.success("Prediction Completed Successfully!")
    st.metric(
        label="Predicted Sale Price",
        value=f"${prediction[0]:,.2f}"
    )

    st.subheader("Entered Details")
    st.write({
        "Overall Quality": OverallQual,
        "Living Area": GrLivArea,
        "Garage Cars": GarageCars,
        "Basement Area": TotalBsmtSF,
        "Full Bathrooms": FullBath,
        "Year Built": YearBuilt,
        "Year Remodeled": YearRemodAdd,
        "Garage Area": GarageArea,
        "First Floor Area": FirstFlrSF,
        "Total Rooms": TotRmsAbvGrd
    })