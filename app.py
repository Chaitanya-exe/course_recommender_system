from recommender_system.eligibility_pipeline import EligibilityRule
from recommender_system.utils.utility import get_string_from_array
from recommender_system.vector_engine import TFIDFEngine, EmbeddingEngine, HybridEngine, VectorEngine
import pandas as pd
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import json
from datetime import datetime

@st.cache_data
def load_data():
    return pd.read_csv("restructure_pipeline/fixed_data.csv")

courses_df = load_data()

if "results" not in st.session_state:
    st.session_state.results = None


@st.cache_resource
def get_db():
    conn = sqlite3.connect('experiment.db', check_same_thread=False)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        student_profile TEXT,
        method TEXT,
        recommended_courses TEXT,
        scores TEXT,
        feedback TEXT
    )
    """)

    conn.commit()
    return conn

@st.cache_resource
def save_to_db(student, data, feedback, method):
    courses, scores = data
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO recommendation_logs
        (timestamp, student_profile, method, recommended_courses, scores, feedback)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        json.dumps(student),
        method,
        json.dumps([c['program_name'] for c in courses]),
        json.dumps(scores),
        feedback
    ))
    conn.commit()
    

@st.cache_resource
def init_engines(corpus):
    tfidf_engine = TFIDFEngine()
    embed_engine = EmbeddingEngine()
    hybrid_engine = HybridEngine(tfidf=tfidf_engine, embedder=embed_engine, alpha=0.3)
    tfidf_engine.fit(corpus)
    embed_engine.fit(corpus)
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

def recommendations_worker(engine: VectorEngine, student, courses_df):
    eligible_courses = []
    for (i, course) in courses_df.iterrows():
        try:
            ok, reason = eligibility.check_eligibility(student=student, course=course)
            if ok:
                eligible_courses.append(course)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    
    st.write(f"eligible for {len(eligible_courses)} courses")
    if not eligible_courses:
        return [], []

    eligible_courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(get_string_from_array(course['skills_learned']))} {" ".join(get_string_from_array(course['career_outcomes']))}""" for course in eligible_courses]
    student_text = f"""{student['academic_background']} {" ".join(student['subjects'])} {" ".join(student['preferred_skills'])} {student['preferred_domain']} {student['career_goal']} {" ".join(student['interests'])} {" ".join(student['preferred_mode'])} {" ".join(student['preferred_duration'])}"""
    course_matrix = engine.transform(eligible_courses_text)
    student_vector = engine.transform([student_text])

    scores = engine.similarity(query_vec=student_vector, matrix=course_matrix)
    top_indices = np.argsort(scores)[::-1][:5]
    recommended_courses = [eligible_courses[i] for i in top_indices]
    final_scores = [scores[i] for i in top_indices]
    return recommended_courses, final_scores
    

if st.button("Get Recommendations"):
    st.spinner("Getting recommendations")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "tfidf": executor.submit(recommendations_worker, tfidf, student, courses_df),
            "embed": executor.submit(recommendations_worker, embed, student, courses_df),
            "hybrid": executor.submit(recommendations_worker, hybrid, student, courses_df)
        }

        results = {}

        for key, future in futures.items():
            results[key] = future.result()
    st.session_state.results = results

if st.session_state.results:
    col1, col2, col3 = st.columns(3)
    def display(col, title, data):
        courses, scores = data
        with col:
            st.write(title)
            for c, s in zip(courses, scores):
                st.write(f"{c['program_name']} {c['duration']} {c['mode']} ({round(s, 3)})")
            
            feedback = st.radio(f"{title} feedback", ["good", "Moderate", "Bad"])
        
        return feedback

    tfidf_feedback = display(col1, "TF-IDF results", st.session_state.results["tfidf"])
    embed_feedback = display(col2, "Embedding results", st.session_state.results["embed"])
    hybrid_feedback = display(col3, "Hybrid results", st.session_state.results["hybrid"])

if st.button("Submit Feedback"):
    st.write("Your feedback")
    st.write([tfidf_feedback, embed_feedback, hybrid_feedback])
    save_to_db(student=student, data=st.session_state.results['tfidf'], feedback=tfidf_feedback, method="tfidf")
    save_to_db(student=student, data=st.session_state.results['embed'], feedback=tfidf_feedback, method="embed")
    save_to_db(student=student, data=st.session_state.results['hybrid'], feedback=tfidf_feedback, method="hybrid")
    st.success("Feedback Submitted")
