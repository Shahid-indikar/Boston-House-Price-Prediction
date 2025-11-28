import streamlit as st
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Train model on startup and cache it
@st.cache_resource
def train_model():
    # Load Boston Housing dataset
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)

    X = df.drop(columns=["medv"])
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return model

model = train_model()

st.title("Boston House Price Prediction App")
st.write("Enter the details below to predict the house price (in $1000s).")

# Input fields
crim = st.number_input("Per capita crime rate (crim)", min_value=0.0, value=0.1, step=0.1)
zn = st.number_input("Residential land zoned over 25,000 sq.ft (zn)", min_value=0.0, value=0.0, step=1.0)
indus = st.number_input("Non-retail business acres per town (indus)", min_value=0.0, value=5.0, step=0.5)
chas = st.selectbox("Tract bounds river (chas)", options=[0, 1], index=0)
nox = st.number_input("Nitric oxides concentration (nox)", min_value=0.0, value=0.5, step=0.01, format="%.3f")
rm = st.number_input("Average number of rooms (rm)", min_value=1.0, value=6.0, step=0.1)
age = st.number_input("Proportion of owner-occupied units built prior to 1940 (age)", min_value=0.0, value=50.0, step=1.0)
dis = st.number_input("Weighted distances to employment centres (dis)", min_value=0.0, value=4.0, step=0.1)
rad = st.number_input("Index of accessibility to radial highways (rad)", min_value=1.0, value=5.0, step=1.0)
tax = st.number_input("Property-tax rate per $10,000 (tax)", min_value=0.0, value=300.0, step=1.0)
ptratio = st.number_input("Pupil–teacher ratio (ptratio)", min_value=1.0, value=18.0, step=0.1)
b = st.number_input("1000(Bk - 0.63)^2 (b)", min_value=0.0, value=350.0, step=1.0)
lstat = st.number_input("% lower status of the population (lstat)", min_value=0.0, value=10.0, step=0.1)

if st.button("Predict Price"):
    input_data = np.array([[crim, zn, indus, chas, nox, rm,
                            age, dis, rad, tax, ptratio, b, lstat]])
    prediction = model.predict(input_data)[0]   # single value
    st.success(f"Estimated House Price: ${prediction * 1000:,.2f}")
