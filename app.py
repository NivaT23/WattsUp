import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from chatbot_module import chatbot_response

# ---------- Page Config ----------
st.set_page_config(page_title="WattsUp", layout="wide")

# ---------- Load ML Model ----------
MODEL_PATH = "bill_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ---------- Page Routing ----------
query_params = st.query_params  
page = query_params.get("page", ["home"])[0] if isinstance(query_params.get("page"), list) else query_params.get("page", "home")

# ---------- HOME PAGE ----------
if page == "home":
    # increase embedded height and allow scrolling so the background image appears larger
    st.components.v1.html(open("index.html", encoding="utf-8").read(), height=920, scrolling=True)


# ---------- BILL PREDICTOR PAGE ----------
elif page == "predict":
    # inside: elif page == "predict":
    # --- inject page-specific background (only for predict page) ---
    bg_url = "https://www.repsol.com/content/dam/repsol-corporate/es/energia-e-innovacion/energia%20electrica%20cables%20alta%20tension.jpg"
    st.markdown(
        f"""
        <style>
        /* make app container background transparent so our ::before shows */
        .stApp {{
            background: transparent;
        }}

        /* add a fixed pseudo-element with the background image + overlay */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            z-index: -1;
            background:
            linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)), /* dark overlay: change opacity here */
            url("{bg_url}") center/cover no-repeat;
            /* tweak these to control "opacity" and blur */
            filter: brightness(0.55) blur(1px); /* blur: 0 (none) -> increase for more blur */
            -webkit-backdrop-filter: blur(1px);
            background-attachment: fixed;
            pointer-events: none;
            transform: translateZ(0);
        }}

        /* optional: make main content card more opaque so text is readable */
        .css-1n76uvr, /* Streamlit main content wrapper (may vary) */
        .main {{
            background: rgba(255,255,255,0.02) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.header("Predict Next Month’s Electricity Bill")
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

# ---------- CHATBOT PAGE ----------
elif page == "chat":

    st.markdown(
        """
        <style>

        /* ===== Background ===== */
        .stApp {
            background:
                linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.75)),
                url("https://www.repsol.com/content/dam/repsol-corporate/es/energia-e-innovacion/energia%20electrica%20cables%20alta%20tension.jpg")
                center/cover no-repeat fixed !important;
        }

        /* ===== Title ===== */
        .chat-title {
            text-align: center;
            color: white;
            font-size: 46px;
            font-weight: 600;
            margin-top: 110px;
            margin-bottom: 5px;
        }

        .chat-sub {
            text-align: center;
            color: #d9f6ff;
            font-size: 18px;
            margin-bottom: 40px;
        }

        /* ===== Chat Bubbles ===== */
        .stChatMessage {
            background: rgba(255,255,255,0.10) !important;
            border-radius: 18px !important;
            padding: 16px !important;
            margin-bottom: 12px !important;
            color: white !important;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.10);
        }

        /* Force all chat text to white */
        .stChatMessage, .stChatMessage * {
            color: white !important;
        }

        /* ===== Remove extra wrapper around input ===== */
        .stChatInput, .stChatInput > div {
            background: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Ensure the input row is a single non-wrapping flex line so the pill doesn't shrink */
        .stChatInput {
            display: flex !important;
            gap: 10px !important;
            align-items: center !important;
            flex-wrap: nowrap !important;
            width: 100% !important;
        }

        /* ===== Neon Text Input ===== */
        .stChatInput textarea {
            background: rgba(15, 20, 30, 0.55) !important;
            color: #e8faff !important;
            border: 1px solid rgba(0,255,255,0.35) !important;
            border-radius: 40px !important;
            padding: 14px 22px !important;
            height: 55px !important;
            font-size: 16px !important;

            box-shadow:
                0 0 12px rgba(0,255,255,0.30),
                inset 0 0 12px rgba(0,255,255,0.20) !important;
            flex: 1 1 auto !important;
            min-width: 0 !important; /* prevents flex item from overflowing and shrinking */
        }

        .stChatInput textarea:focus {
            border: 1px solid rgba(0,255,255,0.65) !important;
            box-shadow:
                0 0 18px rgba(0,255,255,0.75),
                inset 0 0 10px rgba(0,255,255,0.25) !important;
        }

        /* ===== Neon Send Button (flex, scales with input) ===== */
        .stChatInput button {
            background: linear-gradient(135deg, #00eaff, #00ffc8) !important;
            color: black !important;
            font-size: 22px !important;
            font-weight: 900 !important;

            /* allow the button to size with the container while staying visually balanced */
            height: 100% !important; /* match textarea height */
            padding: 0 clamp(10px, 2.2vw, 24px) !important; /* scales with viewport */
            min-width: 48px !important;

            border-radius: 999px !important;
            border: none !important;

            box-shadow:
                0 0 18px rgba(0,255,255,0.75),
                inset 0 0 6px rgba(255,255,255,0.5) !important;

            transition: transform 0.12s ease-in-out, box-shadow 0.12s ease-in-out !important;
            margin-left: 8px !important;
            flex: 0 0 auto !important;
        }

        .stChatInput button:hover {
            transform: scale(1.10);
            box-shadow:
                0 0 26px rgba(0,255,255,1),
                inset 0 0 8px rgba(255,255,255,0.6) !important;
        }

        /* ===== FIX INPUT MISALIGN ===== */
        .stChatInputContainer {
            padding-bottom: 30px !important;
        }

        /* ===== Keep input pill fixed at bottom so it doesn't shrink on rerun ===== */
        .stChatInputContainer {
            position: fixed !important;
            left: 40px !important;
            right: 40px !important;
            bottom: 28px !important;
            z-index: 99990 !important;
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
            padding: 10px 16px !important;
            background: rgba(15,20,30,0.55) !important;
            border-radius: 40px !important;
            border: 1px solid rgba(0,255,255,0.06) !important;
            box-shadow: 0 10px 40px rgba(0,0,0,0.55) !important;
        }

        /* make sure main content doesn't get covered by the fixed input */
        main .block-container {
            padding-bottom: 160px !important;
        }

        @media (max-width: 700px) {
            .stChatInputContainer { left: 14px !important; right: 14px !important; bottom: 18px !important; }
            main .block-container { padding-bottom: 220px !important; }
        }

        /* Extra enforcement rules to prevent the input bar from shrinking on rerun */
        .stChatInputContainer, .stChatInput, .stChatInput > div {
            position: fixed !important;
            left: 40px !important;
            right: 40px !important;
            bottom: 28px !important;
            width: calc(100% - 80px) !important;
            max-width: calc(100% - 80px) !important;
            box-sizing: border-box !important;
            z-index: 999999 !important;
        }

        .stChatInput textarea {
            width: 100% !important;
            flex: 1 1 auto !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
        }

        /* ensure the button stays inline inside the fixed container and scales with it */
        .stChatInput button {
            position: relative !important;
            right: auto !important;
            top: auto !important;
            transform: none !important;
            margin-left: 8px !important;
        }

        /* ===== Top-Left HOME Button ===== */
        .home-btn button {
            position: fixed !important;
            top: 25px !important;
            left: 25px !important;

            background: rgba(255,255,255,0.22) !important;
            color: white !important;
            padding: 8px 20px !important;
            border-radius: 25px !important;
            font-size: 16px !important;

            backdrop-filter: blur(6px) !important;
            border: 1px solid rgba(255,255,255,0.35) !important;

            z-index: 99999 !important;
        }

        .home-btn {
            margin: 0 !important;
            padding: 0 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top-left Home button (Streamlit)
    home_container = st.container()
    with home_container:
        st.markdown('<div class="home-btn">', unsafe_allow_html=True)

        if st.button("← Back to Home"):
            st.query_params["page"] = "home"

        st.markdown('</div>', unsafe_allow_html=True)

    # Title + Subtitle
    st.markdown('<div class="chat-title">WattSon</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-sub">Ask me anything about electricity, bills & energy usage</div>', unsafe_allow_html=True)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past messages
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    # Input
    user_input = st.chat_input("Ask me anything…")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        bot_reply = chatbot_response(user_input)
        st.session_state.chat_history.append(("assistant", bot_reply))

        with st.chat_message("assistant"):
            st.write(bot_reply)
