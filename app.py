import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("best_logistic_regression.pkl")

# Judul Aplikasi
st.title("Customer Churn Prediction")
st.write("""
Aplikasi ini digunakan untuk memprediksi apakah customer berpotensi churn (berhenti menggunakan layanan) atau tidak menggunakan model Logistic Regression.
""")

# Upload File
uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Informasi Dataset")
    st.write(f"Jumlah Data : {len(df)}")
    st.write(f"Jumlah Fitur : {len(df.columns)}")

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    if st.button("Prediksi"):

        # Prediksi
        pred = model.predict(df)

        # Buat DataFrame hasil
        hasil = pd.DataFrame({
            "Prediksi": pred
        })

        # Konversi label
        hasil["Status"] = hasil["Prediksi"].map({
            0: "Tidak Churn",
            1: "Churn"
        })

        st.success("Prediksi Berhasil")

        # Tampilkan hasil
        st.subheader("Hasil Prediksi")
        st.dataframe(hasil.head(20))

        # Ringkasan
        jumlah_churn = (pred == 1).sum()
        jumlah_tidak_churn = (pred == 0).sum()

        st.subheader("Ringkasan Hasil")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Customer Churn",
                jumlah_churn
            )

        with col2:
            st.metric(
                "Customer Tidak Churn",
                jumlah_tidak_churn
            )

        # Visualisasi
        st.subheader("Distribusi Prediksi")

        distribusi = hasil["Status"].value_counts()

        st.bar_chart(distribusi)

        # Download hasil
        csv = hasil.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )