from eligibility_pipeline import EligibilityRule
from utils.utility import get_string_from_array, Finaliser
from vector_engine import TFIDFEngine, EmbeddingEngine, HybridEngine
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_csv("recommender_system/final_data3.csv")

courses_df = load_data()

@st.cache_resource
def init_engines(corpus):
    tfidf_engine = TFIDFEngine()
    embed_engine = EmbeddingEngine()
    hybrid_engine = HybridEngine(tfidf=tfidf_engine, embedder=embed_engine)
    tfidf_engine.fit(corpus)
    return tfidf_engine, embed_engine, hybrid_engine

courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(get_string_from_array(course['skills_learned']))} {" ".join(get_string_from_array(course['career_outcomes']))}""" for (i, course) in courses_df.iterrows()]
tfidf, embed, hybrid = init_engines(courses_text)

eligibility = EligibilityRule()

st.title("🎓 Course Recommendation System")

st.subheader("Student Profile")

degree_level = st.selectbox("Degree Level", ["PreUG", "UG", "PG"])
percentage = st.number_input("Aggregate Percentage", 0, 100)
is_reserved = st.checkbox("Are you from any of the Reserved Category (SC, ST, OBC etc...)")

academic_background = st.text_input("Enter the name of your course ")
subjects = st.text_input("Enter the subjects you have studied in your course (3 to 5 are enough)")
interests = st.text_input("Enter keywords that are in your interests (e.g - AI, Language learning, Poetry etc...)")
preferred_skills = st.text_input("Enter some of the skills that you would like to learn in future.")
preferred_domain = st.selectbox("Preferred Domain", ["Engineering", "Science", "Management", "Commerce", "Humanities"])

if st.button("Get Recommendations"):
    pass

st.subheader("Feedback")

feedback_tfidf = st.radio("TF-IDF Quality", ["Bad", "Moderate", "Good"])
feedback_embed = st.radio("Embedding Quality", ["Bad", "Moderate", "Good"])
feedback_hybrid = st.radio("Hybrid Quality", ["Bad", "Moderate", "Good"])

if st.button("Submit Feedback"):
    st.success("Feedback Submitted")
    pass
