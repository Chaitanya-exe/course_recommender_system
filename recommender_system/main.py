from eligibility_pipeline import EligibilityRule
import pandas as pd
from vectorizer import TFIDFVector, calculate_similarity
import numpy as np
import ast

recommendation_df = pd.DataFrame()
def main():
    students_df = pd.read_csv("recommender_system/students_data.csv")
    courses_df = pd.read_csv("recommender_system/final_data3.csv")
    eligibility_pipe = EligibilityRule()
    tfidf = TFIDFVector()
    courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {ast.literal_eval(course['skills_learned'])} {ast.literal_eval(course['career_outcomes'])}""" for (i, course) in courses_df.iterrows()]
    tfidf.fit(courses_text)

    for (i, student) in students_df.iterrows():
        eligible_courses = []

        for (j, course) in courses_df.iterrows():
            try:
                result, reason = eligibility_pipe.check_eligibility(student, course)

                if result == True:
                    eligible_courses.append(course)
                else:
                    print(reason)
                    continue
            except:
                print(f"Some problem with this course {course["program_name"]}")
                continue
        eligible_courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {ast.literal_eval(course['skills_learned'])} {ast.literal_eval(course['career_outcomes'])}""" for course in eligible_courses]

        print(f"Student {i + 1} eligible for {len(eligible_courses_text)} courses.")
        student_text = f"""{student['academic_background']} {ast.literal_eval(student['subjects'])} {ast.literal_eval(student['preferred_skills'])} {student['preferred_domain']} {student['career_goal']} {ast.literal_eval(student['interests'])}"""
        course_matrix = tfidf.transform(eligible_courses_text)
        student_vector = tfidf.transform([student_text])

        scores = calculate_similarity(student_vector=student_vector, courses_matrix=course_matrix)
        top_indices = np.argsort(scores)[::-1][:7]
        recommended_courses = [eligible_courses[i] for i in top_indices]
        print(f"Student {i+1} was recommended these courses:\n{recommended_courses}")
        print("\n\n\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error occured: ", str(e))
        import traceback
        traceback.print_exc()