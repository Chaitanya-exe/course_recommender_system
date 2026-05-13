from recommender_system import TFIDFEngine, EmbeddingEngine, HybridEngine, EligibilityRule
from recommender_system.utils.utility import get_string_from_array
import pandas as pd
import numpy as np
import json

dataset_path = "dataset/student_profiles_improved.json"
courses_df = pd.read_csv("courses_data.csv")

with open(dataset_path, "r") as file:
    students = json.loads(file.read())
    students_df = pd.DataFrame(students)

def init_engine(corpus):
    tfidf_engine = TFIDFEngine()
    embed_engine = EmbeddingEngine()
    tfidf_engine.fit(corpus)
    hybrid_engine = HybridEngine(tfidf=tfidf_engine, embedder=embed_engine, alpha=0.6)
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
        domain_match = int(c['domain'] == student['preferred_domain'])
        student_subjects = set(student['subjects'])
        course_subjects = set(c['required_subjects'])
        subject_overlap = student_subjects.intersection(course_subjects)

        student_skills = student['preferred_skills']
        course_skills = c['skills_learned']
        skill_relevance = embed_engine.similarity(embed_engine.transform([" ".join(student_skills)]), embed_engine.transform([" ".join(course_skills)]))

        marks_margin = student['percentage'] - (c['min_marks_general'] or c['min_marks_reserved'])

        records.append({
            **c,
            "score": s,
            "domain_match": domain_match,
            "subject_overlap": subject_overlap,
            "marks_margin": marks_margin,
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
                **record,
                "label": None
            })
        print(f"Record processed for student_id: {i+1}")
    
    return pd.DataFrame(results)
    

def main():
    try:
        dataset = build_dataset(students_df=students_df, courses_df=courses_df, engine=hybrid_engine)
        dataset.to_csv("ml_dataset.csv")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error occured :", str(e))

if __name__ == '__main__':
    main()

        

        
           
        
            
            

