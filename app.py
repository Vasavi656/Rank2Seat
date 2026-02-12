import streamlit as st
from model import (
    data,
    predict_probability,
    normalize_college,
    normalize_category,
    normalize_gender
)

st.set_page_config(page_title="Seat Predictor (Logistic Regression)", layout="centered")

st.title("🎓 AP EAMCET Seat Probability Predictor (ML - Logistic Regression)")

rank = st.number_input("Enter Your Rank", min_value=1, step=1)

college_name = st.selectbox(
    "Select College",
    sorted(set(data["CollegeName"].dropna()))
)

branch = st.selectbox(
    "Select Branch",
    sorted(data[data["CollegeName"] == college_name]["Branch"].unique())
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
    sorted(
        data[
            (data["CollegeName"] == college_name) &
            (data["Branch"] == branch) &
            (data["Category"] == category)
        ]["Gender"].unique()
    )
)

if st.button("Predict"):

    probability = predict_probability(
        rank,
        college_name,
        branch,
        category,
        gender
    )

    st.success(f"Admission Probability: {probability}%")

    if probability >= 75:
        st.success("🟢 High chance of admission")
    elif probability >= 40:
        st.warning("🟡 Moderate chance")
    else:
        st.error("🔴 Low chance")
