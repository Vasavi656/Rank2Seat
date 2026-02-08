# app.py
import streamlit as st
import pandas as pd
import os
from model import (
    predict_probability,
    normalize_category,
    normalize_college,
    normalize_gender
)

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(
    page_title="Rank2Seat – AP EAMCET Seat Probability Predictor",
    layout="centered"
)

# ------------------------------------
# TITLE & DESCRIPTION
# ------------------------------------
st.title("🎓 Rank2Seat")
st.subheader("AP EAMCET Seat Probability Predictor")

st.write(
    "Rank2Seat is an **ML-based seat prediction system** that estimates the "
    "**percentage probability of getting a seat** using multi-year AP EAMCET "
    "cutoff trends."
)

# ------------------------------------
# LOAD CSV DATA
# ------------------------------------
CSV_FILES = [
    "sample_data/2019.csv",
    "sample_data/2020.csv",
    "sample_data/2022.csv",
    "sample_data/2023.csv",
    "sample_data/2024.csv",
]

dfs = []
for file in CSV_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        if not df.empty:
            dfs.append(df)

if not dfs:
    st.error("❌ No historical data found")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

# ------------------------------------
# NORMALIZE DATA
# ------------------------------------
data["CollegeName"] = data["CollegeName"].apply(normalize_college)
data["Branch"] = data["Branch"].astype(str).str.upper().str.strip()
data["Gender"] = data["Gender"].apply(normalize_gender)
data["Category"] = data["Category"].apply(normalize_category)

# ------------------------------------
# USER INPUTS
# ------------------------------------
rank = st.number_input(
    "Enter your AP EAMCET Rank",
    min_value=1,
    step=1
)

college_name = st.selectbox(
    "Select College",
    sorted(data["CollegeName"].dropna().unique())
)

college_code = (
    data[data["CollegeName"] == college_name]["CollegeCode"]
    .iloc[0]
)

branch = st.selectbox(
    "Select Branch",
    sorted(
        data[data["CollegeName"] == college_name]["Branch"].unique()
    )
)

category = st.selectbox(
    "Select Category",
    sorted(
        data[
            (data["CollegeName"] == college_name) &
            (data["Branch"] == branch)
        ]["Category"].unique()
    )
)

gender = st.selectbox(
    "Select Gender",
    ["Female", "Male"]
)

# ------------------------------------
# PREDICTION
# ------------------------------------
if st.button("🎯 Predict Probability"):
    prob, avg_cutoff, latest_cutoff, _ = predict_probability(
        rank,
        college_code,
        branch,
        category,
        gender
    )

    if prob is None:
        st.error("❌ No historical data available for this selection")
    else:
        st.success(f"📊 **Probability of getting a seat: {prob}%**")

        st.info(
            f"""
            • **Average cutoff (multi-year):** {avg_cutoff}  
            • **Latest year cutoff:** {latest_cutoff}  
            • **Your rank:** {rank}
            """
        )

        if rank <= latest_cutoff:
            st.success("🟢 Your rank is within the latest cutoff")
        else:
            st.warning("🟡 Your rank is above the latest cutoff")

# ------------------------------------
# FOOTER BRANDING
# ------------------------------------
st.markdown("---")
st.caption("🔍 Rank2Seat – Turning Ranks into Probable Seats")
