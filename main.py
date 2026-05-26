from recommender_system import TFIDFEngine, EmbeddingEngine, HybridEngine, EligibilityRule
from ast import literal_eval
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import json

dataset_path = "dataset/students_data.json"
courses_df = pd.read_csv("dataset/courses_data.csv")

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

def save_progress(
        conn,
        counter:int
):

    cursor=conn.cursor()

    cursor.execute(
    """
    UPDATE progress
    SET counter=?
    WHERE id=1
    """,
    (counter,)
    )

    conn.commit()
    
def load_progress(conn):

    cursor = conn.cursor()

    cursor.execute("""
    SELECT counter
    FROM progress
    WHERE id=1
    """)

    result = cursor.fetchone()

    if result:
        return result[0]

    return 0


def load_students():
    with open(dataset_path, "r") as file:
        students = json.loads(file.read())
        return pd.DataFrame(students)
    
students_df = load_students()

labels = {
    "Good": 2,
    "Moderate": 1,
    "Bad": 0
}

@st.cache_resource
def get_db():
    conn = sqlite3.connect('experiment.db', check_same_thread=False)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,

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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        counter INTEGER
    )
    """)

    cursor.execute("""
    INSERT OR IGNORE INTO progress(id, counter)
    VALUES(1,0)
    """)

    conn.commit()
    return conn

def save_results(
        conn: sqlite3.Connection,
        student_id: int,
        student,
        recommendations
):

    cursor = conn.cursor()

    for rec in recommendations:

        cursor.execute(
        """
        SELECT id
        FROM dataset
        WHERE student_id=?
        AND program_name=?
        """,
        (
            student_id,
            rec["program_name"]
        )
        )

        existing=cursor.fetchone()

        if existing:
            print("Record already exists!!")
            continue

        cursor.execute(
        """
        INSERT INTO dataset(

        student_id,

        degree_level,
        percentage,
        is_reserved,

        academic_background,

        subjects,
        interests,
        career_goal,
        preferred_skills,

        preferred_domain,
        preferred_duration,
        preferred_mode,

        program_name,
        program_level,
        domain,
        duration,
        mode,
        description,

        skills_learned,
        career_outcomes,

        eligibility,

        required_subjects,

        min_degree_level,

        min_marks_general,
        min_marks_reserved,

        similarity_score,

        domain_match,
        subject_required,
        subject_overlap,

        marks_required,
        marks_margin,

        skill_relevance_percentage,
        career_align_percentage,

        label

        )

        VALUES(

        ?,?,?,?,?,?,
        ?,?,?,?,?,?,
        ?,?,?,?,?,?,
        ?,?,?,?,?,?,
        ?,?,?,?,?,?,
        ?,?,?,?

        )
        """,

        (

        student_id,

        student["degree_level"],
        float(student["percentage"]),
        int(student["is_reserved"]),

        student["academic_background"],

        json.dumps(
            student["subjects"]
        ),

        json.dumps(
            student["interests"]
        ),

        student["career_goal"],

        json.dumps(
            student["preferred_skills"]
        ),

        student["preferred_domain"],
        student["preferred_duration"],
        student["preferred_mode"],

        rec["program_name"],
        rec["program_level"],
        rec["domain"],
        rec["duration"],
        rec["mode"],
        rec["description"],

        json.dumps(
            rec["skills_learned"]
        ),

        json.dumps(
            rec["career_outcomes"]
        ),

        rec["eligibility"],

        json.dumps(
            rec["required_subjects"]
        ),

        rec["min_degree_level"],

        rec["min_marks_general"],
        rec["min_marks_reserved"],

        float(rec["score"]),

        int(rec["domain_match"]),
        int(rec["subject_required"]),
        int(rec["subject_overlap"]),

        int(rec["marks_required"]),

        float(rec["marks_margin"]),

        float(
            rec["skill_relevance_percentage"]
        ),

        float(
            rec["career_alignment_percentage"]
        ),

        int(rec["label"])

        )
        )

    conn.commit()


if "progress" not in st.session_state:
    st.session_state.progress = load_progress(conn=get_db())

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
        course_skills = literal_eval(c['skills_learned'])
        skill_relevance_percentage = embed_engine.similarity(embed_engine.transform([" ".join(student_skills)]), embed_engine.transform([" ".join(course_skills)])) * 100

        student_vec = embed_engine.transform([student['career_goal']])
        career_outcomes = literal_eval(c['career_outcomes'])
        career_alignment_percentage = embed_engine.similarity(student_vec, embed_engine.transform([" ".join(career_outcomes)])) * 100

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
        
student = students_df.iloc[st.session_state.progress]

st.header("Data Annotation")
st.subheader("Course Recommendation system")

st.subheader("Student Profile")
with st.container():

    st.markdown("### Profile")

    c1,c2=st.columns(2)

    with c1:

        st.write(
            f"**Background:** {student['academic_background']}"
        )

        st.write(
            f"**Degree:** {student['degree_level']}"
        )

        st.write(
            f"**Percentage:** {student['percentage']}"
        )

        st.write(
            f"**Domain:** {student['preferred_domain']}"
        )

        st.write(
            f"**Career Goal:** {student['career_goal']}"
        )


    with c2:

        st.write(
            f"**Subjects:** {', '.join(student['subjects'])}"
        )

        st.write(
            f"**Interests:** {', '.join(student['interests'])}"
        )

        st.write(
            f"**Skills:** {', '.join(student['preferred_skills'])}"
        )

        st.write(
            f"**Mode:** {student['preferred_mode']}"
        )

        st.write(
            f"**Duration:** {student['preferred_duration']}"
        )

if st.button("Get Recommendations"):

    eligible_courses = []
    
    for i, course in courses_df.iterrows():
        try:
            ok, _ = eligibility.check_eligibility(student=student, course=course)
            if ok:
                eligible_courses.append(course)
            else:
                continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Some error occured: ", str(e))
            continue
    
    st.session_state.recommendations = extract_features(student=student, courses=eligible_courses, hybrid=hybrid_engine) 


if st.session_state.recommendations is not None:
    for idx,course in enumerate(
        st.session_state.recommendations
    ):

        with st.container():

            st.markdown(
            f"""
            ### {course['program_name']}

            **Score:** {round(course['score'],3)}

            **Domain:** {course['domain']}

            **Career Outcomes:** {course['career_outcomes']}

            **Skills learned:** {course['skills_learned']}

            **Career Alignment:** 
            {round(course['career_alignment_percentage'],1)}%
            
            **Skill Relevance:** 
            {round(course['skill_relevance_percentage'],1)}%
            
            **required subjects:** {course['required_subjects']}

            **Subject Required:** {course['subject_required']}

            **Subject Overlap:** 
            {course['subject_overlap']}

            **Domain match: ** {course['domain_match']}
            
            **Description**
            {course['description']}
            """
            )


            selected=st.selectbox(

                "Recommendation Quality",

                [

                "Good",
                "Moderate",
                "Bad"

                ],

                key=f"label_{idx}"

            )

            course["label"]=labels[
                selected
            ]

            st.markdown("---")

save_button = st.button("Save to DB")

if save_button:
    try:
        save_results(conn=get_db(), student_id=st.session_state.progress, student=student, recommendations=st.session_state.recommendations)
        st.success("Saved to DB")
        st.session_state.progress += 1
        st.session_state.recommendations = None
        st.rerun()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error saving to DB: ", str(e))

if st.button("Store Progress"):
    save_progress(conn=get_db(), counter=st.session_state.progress)
    st.success("Progress saved successfully")