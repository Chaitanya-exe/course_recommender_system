from recommender_system import TFIDFEngine, EmbeddingEngine, HybridEngine, EligibilityRule
import sqlite3
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json

dataset_path = "students_data.json"
courses_df = pd.read_csv("courses_data.csv")

@st.cache_resource
def retreive_model():
    return joblib.load("models/rf_model.pkl")

@st.cache_resource
def init_engines(corpus):
    tfidf_engine = TFIDFEngine()
    embed_engine = EmbeddingEngine()
    hybrid_engine = HybridEngine(tfidf=tfidf_engine, embedder=embed_engine, alpha=0.3)
    tfidf_engine.fit(corpus)
    embed_engine.fit(corpus)
    return tfidf_engine, embed_engine, hybrid_engine

courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(course['skills_learned'])} {" ".join(course['career_outcomes'])} {course['mode']} {course['duration']}""" for (i, course) in courses_df.iterrows()]
tfidf, embed, hybrid = init_engines(courses_text)

eligibility = EligibilityRule()

st.title("🎓 Course Recommendation System")

st.subheader("Student Profile")

degree_level = st.selectbox("Degree Level", ["PreUG", "UG", "PG"])
percentage = st.number_input("Aggregate Percentage", 0, 100)
is_reserved = st.checkbox("Are you from any of the Reserved Category (SC, ST, OBC etc...)")

academic_background = st.text_input("Enter the name of your course ")
subjects = st.multiselect("Enter the subjects you have studied in your course (3 to 5 are enough)", [], max_selections=5, accept_new_options=True)
interests = st.multiselect("Enter keywords that best describes your interests", [], max_selections=5, accept_new_options=True)
career_goal = st.text_input("In a short sentence, tell us your career goal")
preferred_skills = st.multiselect("Enter some of the skills that you would like to learn in future.", [], accept_new_options=True)
preferred_domain = st.multiselect("Preferred Domain", pd.unique(courses_df["domain"]), max_selections=1, accept_new_options=False)
preferred_mode = st.multiselect("Preferred mode of Studying", ["Regular", "Distance Learning", "Part Time"], accept_new_options=False) 
preferred_duration = st.multiselect("Preferred duraiton of the course", [], placeholder="enter your preferred duration. 2 year, 4 year, few months etc...", accept_new_options=True)

student = {
    "academic_background": academic_background,
    "degree_level": degree_level,
    "percentage": percentage,
    "is_reserved": is_reserved,
    "subjects": subjects,
    "interests": interests,
    "career_goal": career_goal,
    "preferred_skills": preferred_skills,
    "preferred_domain": preferred_domain,
    "preferred_mode": preferred_mode,
    "preferred_duration": preferred_duration
}