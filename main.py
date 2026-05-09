from concurrent.futures import ThreadPoolExecutor
from recommender_system import VectorEngine, TFIDFEngine, EmbeddingEngine, HybridEngine, EligibilityRule
from recommender_system.utils.utility import get_string_from_array
import pandas as pd
import numpy as np
import json

dataset_path = "../dataset/student_profiles_improved.json"
courses_df = pd.read_csv("fixed_data.csv")

with open(dataset_path, "r") as file:
    students = json.loads(file.read())
    students_df = pd.DataFrame(students)

def init_engines(corpus):
    tfidf_engine = TFIDFEngine()
    embed_engine = EmbeddingEngine()
    hybrid_engine = HybridEngine(tfidf=tfidf_engine, embedder=embed_engine, alpha=0.6)
    tfidf_engine.fit(corpus)
    embed_engine.fit(corpus)
    return tfidf_engine, embed_engine, hybrid_engine

tfidf_engine, embed_engine, hybrid_engine = init_engines(corpus=courses_df)

eligibility = EligibilityRule()

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
    
    print(f"eligible for {len(eligible_courses)} courses")
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
    return final_scores, recommended_courses



def main():
    results = []
    for i, student in students_df.iterrows():
        scores = {}
        with ThreadPoolExecutor(max_workers=3, ) as executor:
            futures = {
                "tfidf": executor.submit(recommendations_worker, tfidf_engine, student, courses_df),
                "embed": executor.submit(recommendations_worker, embed_engine, student, courses_df),
                "hybrid": executor.submit(recommendations_worker, hybrid_engine, student, courses_df)
            }


            for key, value in futures.items():
                scores[key] = value
                
        for key, value in scores.items():
            for c, s in zip(value):
                record = {

                }

        

if __name__ == '__main__':
    main()

        

        
           
        
            
            

