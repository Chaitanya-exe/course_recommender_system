from recommender_system import TFIDFEngine, EmbeddingEngine, HybridEngine, EligibilityRule
import sqlite3
import streamlit as st
from ast import literal_eval
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

button = st.button("Try for recommendations?")

def extract_features(student, courses, hybrid):
    
    
    course_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(course['skills_learned'])} {" ".join(course['career_outcomes'])} {course['mode']} {course['duration']}""" for course in courses]
    student_text = f"""{student['academic_background']} {student['preferred_domain']} {" ".join(student['interests'])} {" ".join(student['subjects'])} {" ".join(student['preferred_skills'])}  {student['career_goal']}  {" ".join(student['preferred_mode'])} {" ".join(student['preferred_duration'])}"""

    scores = hybrid.similarity(
        hybrid.transform(course_text),
        hybrid.transform([student_text])
    )

    top_k = np.argsort(scores)[::-1][:6]
    recommended_courses = [courses[i] for i in top_k]
    final_scores = [scores[i] for i in top_k]
    records = []

    for c, s in zip(recommended_courses, final_scores):
        marks_required = True
        subjects_required = True
        student_subjects = set([subj.lower() for subj in student['subjects']])
        required_subjects = literal_eval(c['required_subjects'])
        course_subjects = set([subj.lower() for subj in required_subjects])

        if len(course_subjects) == 0:
            subjects_required = False
        
        if pd.isna(c['min_marks_general']) and pd.isna(c['min_marks_reserved']):
            marks_required = False
        
        subject_overlap = len(student_subjects.intersection(course_subjects))

        domain_match = int(c['domain'] == student['preferred_domain'])
        student_skills = student['preferred_skills']
        course_skills = literal_eval(c['skills_learned'])
        skill_relevance_percentage = embed.similarity(embed.transform([" ".join(student_skills)]), embed.transform([" ".join(course_skills)])) * 100

        student_vec = embed.transform([student['career_goal']])
        career_outcomes = literal_eval(c['career_outcomes'])
        career_alignment_percentage = embed.similarity(student_vec, embed.transform([" ".join(career_outcomes)])) * 100

        marks_margin = student['percentage'] - (c['min_marks_general'] or c['min_marks_reserved'])

        records.append({
            **c,
            "score": s,
            "domain_match": domain_match,
            "subject_required": subjects_required,
            "subject_overlap": subject_overlap,
            "marks_required": marks_required,
            "marks_margin": marks_margin,
            "skill_relevance_percentage": skill_relevance_percentage[0],
            "career_alignment_percentage": career_alignment_percentage[0]
        })
    
    return records


if button:
    eligible = []
    for i, course in courses_df.iterrows():
        try:
            ok, _ = eligibility.check_eligibility(student=student, course=course)
            
            if ok:
                eligible.append(course)
            else:
                continue
        except Exception as e:
            print("Some error occured: ", e)
            continue

    X = pd.DataFrame()


    
