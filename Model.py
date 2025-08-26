# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 09:56:31 2025

@author: Kirti
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# -------------------------------
# Streamlit Setup
# -------------------------------
st.set_page_config(page_title="Blood Parameters Predictor", layout="wide")
st.title("🩸 Virtual Blood Test Lab")
st.caption("A smart assistant for predicting **Glucose, Hematocrit, Hemoglobin, Oxygen Occupancy** with health insights.")

# -------------------------------
# Disease & Recommendation Logic
# -------------------------------
def interpret_results(gc, hv, hgb, occ):
    diseases = []
    recommendation = []
    status = "✅ Healthy"

    # Glucose
    if gc > 126:
        diseases.append("Hyperglycemia / Diabetes risk")
        recommendation.append("Endocrinology consult, HbA1c test")
        status = "⚠️ Abnormal"
    elif gc < 70:
        diseases.append("Hypoglycemia")
        recommendation.append("Immediate sugar intake, check insulin levels")
        status = "⚠️ Abnormal"

    # Hematocrit
    if hv < 36:
        diseases.append("Low Hematocrit (Anemia)")
        recommendation.append("Iron studies, hematology consult")
        status = "⚠️ Abnormal"
    elif hv > 50:
        diseases.append("High Hematocrit (Polycythemia)")
        recommendation.append("Blood viscosity check, possible oxygen therapy")
        status = "⚠️ Abnormal"

    # Hemoglobin
    if hgb < 12:
        diseases.append("Low Hemoglobin")
        recommendation.append("Iron supplements, dietary improvements")
        status = "⚠️ Abnormal"
    elif hgb > 17.5:
        diseases.append("High Hemoglobin")
        recommendation.append("Check lung/heart function")
        status = "⚠️ Abnormal"

    # Oxygen Occupancy
    if occ < 90:
        diseases.append("Low Oxygen Saturation (Hypoxemia)")
        recommendation.append("Pulmonary function test, oxygen support")
        status = "⚠️ Abnormal"

    if not diseases:
        diseases.append("No abnormality detected 🎉")
        recommendation.append("Routine annual checkup only")

    return "; ".join(diseases), "; ".join(recommendation), status

# -------------------------------
# Training Function
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df[['Ip', 'Tp']]
    y = df[['Gc', 'Hv', 'Hgb', 'Occ']]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model.fit(X_train, y_train)

    return model, scaler_X, scaler_y, X_test, y_test

# -------------------------------
# Input Selection
# -------------------------------
option = st.radio("Choose input method:", ["📂 Upload CSV", "✍️ Enter Manually"])

# -------------------------------
# CSV Upload Mode
# -------------------------------
if option == "📂 Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        model, scaler_X, scaler_y, X_test, y_test = train_model(df)

        # Predictions
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        pred_df = pd.DataFrame(y_pred, columns=['Gc_pred', 'Hv_pred', 'Hgb_pred', 'Occ_pred'])

        # Add interpretations
        pred_df["Diseases_Detected"], pred_df["Checkup_Recommendation"], pred_df["Status"] = zip(
            *pred_df.apply(lambda row: interpret_results(
                row["Gc_pred"], row["Hv_pred"], row["Hgb_pred"], row["Occ_pred"]), axis=1)
        )

        st.subheader("✅ Predictions with Health Insights")
        st.dataframe(pred_df.head())

        # Download option
        st.download_button(
            "📥 Download Full Predictions",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions_with_insights.csv",
            mime="text/csv"
        )

# -------------------------------
# Manual Input Mode
# -------------------------------
elif option == "✍️ Enter Manually":
    st.subheader("🔮 Manual Input Prediction")

    # Load base dataset
    df = pd.read_csv("Original_data_Hgb_o2_vis_saved_.csv")
    model, scaler_X, scaler_y, _, _ = train_model(df)

    Ip_val = st.number_input("Enter Ip value", value=0.0)
    Tp_val = st.number_input("Enter Tp value", value=0.0)

    if st.button("Predict Now"):
        user_input = scaler_X.transform([[Ip_val, Tp_val]])
        user_pred_scaled = model.predict(user_input)
        user_pred = scaler_y.inverse_transform(user_pred_scaled)

        gc, hv, hgb, occ = user_pred[0]
        diseases, recommendation, status = interpret_results(gc, hv, hgb, occ)

        st.markdown(f"### Prediction Status: **{status}**")
        st.write(f"- **Glucose (Gc):** {gc:.2f}")
        st.write(f"- **Hematocrit (Hv):** {hv:.2f}")
        st.write(f"- **Hemoglobin (Hgb):** {hgb:.2f}")
        st.write(f"- **Oxygen Occupancy (Occ):** {occ:.2f}")

        st.subheader("🩺 Doctor’s Notes")
        with st.expander("See Detailed Health Insights"):
            st.write(f"**Diseases Detected:** {diseases}")
            st.write(f"**Checkup Recommendation:** {recommendation}")

        # Radar Chart for visualization
        radar_df = pd.DataFrame({
            "Parameter": ["Glucose", "Hematocrit", "Hemoglobin", "Oxygen Occupancy"],
            "Value": [gc, hv, hgb, occ]
        })
        fig = px.line_polar(radar_df, r="Value", theta="Parameter", line_close=True)
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

