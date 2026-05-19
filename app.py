import pandas as pd
import streamlit as st
import plotly.express as px

st.title("LLM Safety Evaluation Dashboard")

df = pd.read_csv("../results/results.csv")

st.subheader("Evaluation Results")

st.dataframe(df)

st.subheader("Jailbreak Success Rate by Category")

category_scores = (
    df.groupby("category")["jailbreak_success"]
    .mean()
    .reset_index()
)

fig = px.bar(
    category_scores,
    x="category",
    y="jailbreak_success",
    title="Jailbreak Success Rate"
)

st.plotly_chart(fig)

st.subheader("Refusal Distribution")

refusal_counts = (
    df["refusal_detected"]
    .value_counts()
    .reset_index()
)

refusal_counts.columns = ["refusal_detected", "count"]

fig2 = px.pie(
    refusal_counts,
    names="refusal_detected",
    values="count",
    title="Refusal Behavior"
)

st.plotly_chart(fig2)
