import streamlit as st
from src.predict_bert import EssayScorer

scorer = EssayScorer()

st.set_page_config(page_title="AI Essay Grader", layout="centered")
st.title("📝 AI Essay Grading System")

essay_input = st.text_area("Enter your essay below:", height=300)

if st.button("Grade Essay"):
    if essay_input.strip():
        with st.spinner("Evaluating..."):
            score = scorer.predict(essay_input)
        st.success(f"Predicted Score: {score}/10")
    else:
        st.warning("Please enter an essay before grading.")
