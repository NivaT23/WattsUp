import streamlit as st

st.set_page_config(page_title="AI Energy Advisor", layout="centered")
st.components.v1.html(open("index.html").read(), height=700, scrolling=False)