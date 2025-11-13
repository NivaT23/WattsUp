import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- Page Config ----------
st.set_page_config(page_title="WattsUp", layout="centered")

# ---------- Load ML Model ----------
MODEL_PATH = "bill_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ---------- Page Routing ----------
query_params = st.query_params  
page = query_params.get("page", ["home"])[0] if isinstance(query_params.get("page"), list) else query_params.get("page", "home")

# ---------- HOME PAGE ----------
if page == "home":
    st.components.v1.html(open("index.html").read(), height=700, scrolling=False)

# ---------- BILL PREDICTOR PAGE ----------
elif page == "predict":
    st.header("Predict Next Month’s Electricity Bill")
    st.markdown('<style>.stApp{background:url("https://www.repsol.com/content/dam/repsol-corporate/es/energia-e-innovacion/energia%20electrica%20cables%20alta%20tension.jpg") center/cover}</style>', unsafe_allow_html=True)
    st.markdown(
        "Enter your last 3 months' Units and Bills (₹). "
        "The model estimates the next month’s consumption and predicts the bill."
    )

    col_inputs, col_chart = st.columns([1, 2])

    with col_inputs:
        st.subheader("Last 3 months")
        u3 = st.number_input("Units (3 months ago)", min_value=0, value=200, step=1)
        b3 = st.number_input("Bill (3 months ago) ₹", min_value=0, value=1200, step=10)
        u2 = st.number_input("Units (2 months ago)", min_value=0, value=220, step=1)
        b2 = st.number_input("Bill (2 months ago) ₹", min_value=0, value=1320, step=10)
        u1 = st.number_input("Units (last month)", min_value=0, value=240, step=1)
        b1 = st.number_input("Bill (last month) ₹", min_value=0, value=1440, step=10)

        if st.button("Predict Bill"):
            if model is None:
                st.error("No trained model found. Run `train_model.py` first.")
            else:
                try:
                    diffs = [u2 - u3, u1 - u2]
                    avg_diff = np.mean(diffs)
                    next_units = max(0, int(round(u1 + avg_diff)))
                except Exception:
                    next_units = u1

                predicted_bill = float(model.predict([[next_units]])[0])
                st.session_state["pred_val"] = round(predicted_bill, 2)
                st.session_state["pred_units"] = int(next_units)

    with col_chart:
        if "pred_val" in st.session_state:
            months = ["3-mo", "2-mo", "Last", "Next (pred)"]
            bills = [b3, b2, b1, st.session_state.pred_val]
            units = [u3, u2, u1, st.session_state.pred_units]

            st.subheader("Bill (₹) — Actual vs Predicted")
            st.line_chart(pd.DataFrame({"Month": months, "Bill (₹)": bills}).set_index("Month"))

            st.subheader("Units — Actual vs Predicted")
            st.line_chart(pd.DataFrame({"Month": months, "Units": units}).set_index("Month"))

            st.success(
                f"Predicted next bill: ₹{st.session_state.pred_val} "
                f"(Predicted units: {st.session_state.pred_units})"
            )
        else:
            st.info("Enter values and click **Predict Bill** to generate predictions.")

    # Back button
    if st.button("← Back to Home"):
        st.query_params["page"] = "home"

