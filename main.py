from recommender_system import TFIDFEngine, EmbeddingEngine, HybridEngine, EligibilityRule
from ast import literal_eval
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import json

dataset_path = "dataset/students_data.json"
courses_df = pd.read_csv("dataset/courses_data.csv")

if "progress" not in st.session_state:
    st.session_state.progress = 0

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

@st.cache_data
def load_students():
    with open(dataset_path, "r") as file:
        students = json.loads(file.read())
        return pd.DataFrame(students)
    
students_df = load_students()

@st.cache_resource
def get_db():
    conn = sqlite3.connect('experiment.db', check_same_thread=False)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        course_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        degree_level TEXT,
        percentage REAL,
        is_reserved INTEGER,
        academic_background TEXT,
        subjects TEXT,
        interests TEXT,
        career_goal TEXT,
        preferred_skills TEXT,
        preferred_domain TEXT,
        preferred_duration TEXT,
        preferred_mode TEXT,
        
        program_name TEXT,
        program_level TEXT,
        domain TEXT,
        duration TEXT,
        mode TEXT,
        description TEXT,
        skills_learned TEXT,
        career_outcomes TEXT,
        eligibility TEXT,
        required_subjects TEXT,
        min_degree_level TEXT,
        min_marks_general REAL,
        min_marks_reserved REAL,
                   
        similarity_score REAL,
                   
        domain_match INTEGER,
        subject_required INTEGER,
        subject_overlap INTEGER,
        marks_required INTEGER,
        marks_margin INTEGER,
        skill_relevance_percentage REAL,
        career_align_percentage REAL,

        label INTEGER     
    )
    """)

    conn.commit()
    return conn

@st.cache_resource
def init_engine(corpus):
    tfidf_engine = TFIDFEngine()
    embed_engine = EmbeddingEngine()
    tfidf_engine.fit(corpus)
    hybrid_engine = HybridEngine(tfidf=tfidf_engine, embedder=embed_engine, alpha=0.65)
    return hybrid_engine, embed_engine, tfidf_engine

courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(course['skills_learned'])} {" ".join(course['career_outcomes'])} {course['mode']} {course['duration']}""" for (i, course) in courses_df.iterrows()]
hybrid_engine, embed_engine, tfidf_engine = init_engine(corpus=courses_text)
eligibility = EligibilityRule()

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
        course_skills = c['skills_learned']
        skill_relevance_percentage = embed_engine.similarity(embed_engine.transform([" ".join(student_skills)]), embed_engine.transform([" ".join(course_skills)])) * 100

        student_vec = embed_engine.transform([student['career_goal']])

        career_alignment_percentage = max(
            embed_engine.similarity(
                student_vec,
                embed_engine.transform([outcome])
            )
            for outcome in c['career_outcomes']
        ) * 100

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
        

def build_dataset(students_df, courses_df, engine) -> pd.DataFrame:
    results = []
    for i, student in students_df.iterrows():
        eligible_courses = []
        for j, course in courses_df.iterrows():
            try:
                ok, reason = eligibility.check_eligibility(student=student, course=course)
                if ok:
                    eligible_courses.append(course)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error occured: ", str(e))
            
        records = extract_features(student=student, courses=eligible_courses, hybrid=engine)

        for record in records:
            results.append({
                "student_id": i+1,
                **student,
                **record,
                "label": None
            })
        print(f"Record processed for student_id: {i+1}")
    
    return pd.DataFrame(results)

student = students_df.iloc[st.session_state.progress]

st.header("Data Annotation")
st.subheader("Course Recommendation system")


left, right = st.columns([1, 1])

with left:

    st.subheader("Student Profile")

    st.write(
        {
            "Background":
            student['academic_background'],

            "Subjects":
            student['subjects'],

            "Interests":
            student['interests'],

            "Career":
            student['career_goal'],

            "Skills":
            student['preferred_skills'],

            "Domain":
            student['preferred_domain']
        }
    )

with right:
    st.subheader("recommendations")

    if st.button("Get Recommendations"):
        st.write(
            {
                "Recommendations": "some recommendations here"
            }
        )


save_button = st.button("Save to DB")

if save_button:
    st.success("Saved to DB")
    st.session_state.progress += 1
    st.rerun()
