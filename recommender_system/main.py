from eligibility_pipeline import EligibilityRule
from utils.utility import get_string_from_array, Finaliser
from vector_engine import TFIDFEngine, EmbeddingEngine, HybridEngine
import pandas as pd
import numpy as np
import argparse

recommendation_df = pd.DataFrame()
def tfidf_test():

    students_df = pd.read_csv("recommender_system/students_data.csv")
    courses_df = pd.read_csv("recommender_system/courses_data.csv")
    tfidf = TFIDFEngine()
    tfidf_elig = TFIDFEngine()
    eligibility_text = [text for text in courses_df['eligibility']]
    tfidf_elig.fit(corpus=eligibility_text)
    eligibility_pipe = EligibilityRule(vectoriser=tfidf_elig)
    courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(get_string_from_array(course['skills_learned']))} {" ".join(get_string_from_array(course['career_outcomes']))}""" for (i, course) in courses_df.iterrows()]
    tfidf.fit(courses_text)
    finaliser = Finaliser()

    for (i, student) in students_df.iterrows():
        eligible_courses = []

        for (j, course) in courses_df.iterrows():
            try:
                result, reason = eligibility_pipe.check_eligibility(student, course)

                if result == True:
                    eligible_courses.append(course)
                    print(f"{course['program_name']}\n{course['eligibility_keywords']}")
                else:
                    continue
            except:
                print(f"Some problem with this course {course["program_name"]}")
                continue
        eligible_courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(get_string_from_array(course['skills_learned']))} {" ".join(get_string_from_array(course['career_outcomes']))}""" for course in eligible_courses]

        print(f"Student {i + 1} eligible for {len(eligible_courses_text)} courses.")
        student_text = f"""{student['academic_background']} {" ".join(get_string_from_array(student['subjects']))} {" ".join(get_string_from_array(student['preferred_skills']))} {student['preferred_domain']} {student['career_goal']} {" ".join(get_string_from_array(student['interests']))}"""
        course_matrix = tfidf.transform(eligible_courses_text)
        student_vector = tfidf.transform([student_text])

        scores = tfidf.similarity(query_vec=student_vector, matrix=course_matrix)
        top_indices = np.argsort(scores)[::-1][:5]
        recommended_courses = [eligible_courses[i] for i in top_indices]
        final_scores = [scores[i] for i in top_indices]
        finaliser.add_record(student_id=i, recommendations=recommended_courses, algorithm="TF-IDF", student_domain=student['preferred_domain'], scores=final_scores)
        print("\n\n\n")
    
    finaliser.to_csv(filename="tfidf_results.csv")
    print(f"Results has been saved to a csv file in the current directory.")

def embedding_test():
    
    students_df = pd.read_csv("recommender_system/students_data.csv")
    courses_df = pd.read_csv("recommender_system/courses_data.csv")
    engine = EmbeddingEngine()
    elig_engine = EmbeddingEngine()
    eligibility_text = [text for text in courses_df['eligibility']]
    elig_engine.fit(corpus=eligibility_text)
    eligibility_pipe = EligibilityRule(vectoriser=elig_engine)
    finaliser = Finaliser()

    for (i, student) in students_df.iterrows():
        eligible_courses = []

        for (j, course) in courses_df.iterrows():
            try:
                result, reason = eligibility_pipe.check_eligibility(student, course)

                if result == True:
                    eligible_courses.append(course)
                    print(f"{course['program_name']}")
                else:
                    continue
            except:
                print(f"Some problem with this course {course["program_name"]}")
                continue
        eligible_courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(get_string_from_array(course['skills_learned']))} {" ".join(get_string_from_array(course['career_outcomes']))}""" for course in eligible_courses]

        print(f"Student {i + 1} eligible for {len(eligible_courses_text)} courses.")
        engine.fit(corpus=eligible_courses_text)
        student_text = f"""{student['academic_background']} {" ".join(get_string_from_array(student['subjects']))} {" ".join(get_string_from_array(student['preferred_skills']))} {student['preferred_domain']} {student['career_goal']} {" ".join(get_string_from_array(student['interests']))}"""

        student_vector = engine.transform([student_text])

        scores = engine.similarity(query_vec=student_vector)
        top_indices = np.argsort(scores)[::-1][:5]
        recommended_courses = [eligible_courses[i] for i in top_indices]
        final_scores = [scores[i] for i in top_indices]
        finaliser.add_record(student_id=i, recommendations=recommended_courses, algorithm="Embeddings", student_domain=student['preferred_domain'], scores=final_scores)
        print("\n\n\n")
    
    finaliser.to_csv(filename="embedding_test.csv")
    print(f"Results has been saved to a csv file in the current directory.")


def hybrid_test():
    students_df = pd.read_csv("recommender_system/students_data.csv")
    courses_df = pd.read_csv("recommender_system/courses_data.csv")
    engine = HybridEngine()
    elig_engine = HybridEngine()
    eligibility_text = [text for text in courses_df['eligibility']]
    elig_engine.fit(corpus=eligibility_text)
    eligibility_pipe = EligibilityRule(vectoriser=elig_engine)
    courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(get_string_from_array(course['skills_learned']))} {" ".join(get_string_from_array(course['career_outcomes']))}""" for (i, course) in courses_df.iterrows()]
    engine.fit(corpus=courses_text)
    finaliser = Finaliser()

    for (i, student) in students_df.iterrows():
        eligible_courses = []

        for (j, course) in courses_df.iterrows():
            try:
                result, reason = eligibility_pipe.check_eligibility(student, course)

                if result == True:
                    eligible_courses.append(course)
                    print(f"{course['program_name']}")
                else:
                    continue
            except:
                print(f"Some problem with this course {course["program_name"]}")
                continue
        eligible_courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(get_string_from_array(course['skills_learned']))} {" ".join(get_string_from_array(course['career_outcomes']))}""" for course in eligible_courses]

        print(f"Student {i + 1} eligible for {len(eligible_courses_text)} courses.")
        engine.fit(corpus=eligible_courses_text)
        student_text = f"""{student['academic_background']} {" ".join(get_string_from_array(student['subjects']))} {" ".join(get_string_from_array(student['preferred_skills']))} {student['preferred_domain']} {student['career_goal']} {" ".join(get_string_from_array(student['interests']))}"""

        student_vector = engine.transform([student_text])

        scores = engine.similarity(query_vec=student_vector)
        top_indices = np.argsort(scores)[::-1][:5]
        recommended_courses = [eligible_courses[i] for i in top_indices]
        final_scores = [scores[i] for i in top_indices]
        finaliser.add_record(student_id=i, recommendations=recommended_courses, algorithm="Embeddings", student_domain=student['preferred_domain'], scores=final_scores)
        print("\n\n\n")
    
    finaliser.to_csv(filename="embedding_test.csv")
    print(f"Results has been saved to a csv file in the current directory.")
    

def main():
    embedding_test()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error occured: ", str(e))
        import traceback
        traceback.print_exc()