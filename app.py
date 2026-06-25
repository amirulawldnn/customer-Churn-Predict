import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_logistic_regression.pkl")

st.title("Customer Churn Prediction")

uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.write("Jumlah Data:")
    st.write(len(df))

    if st.button("Prediksi"):

        pred = model.predict(df)

        st.success("Prediksi Berhasil")

        st.write(pred[:20])