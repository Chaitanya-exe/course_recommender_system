from eligibility_pipeline import EligibilityRule
import pandas as pd
import re
from vectorizer import TFIDFVector, calculate_similarity

def parse_eligibility_struct(text):
    degree = re.search(r"min_degree_level='(.*?)'", text)
    gen = re.search(r"min_marks_general=([\d.]+)", text)
    res = re.search(r"min_marks_reserved=([\d.]+)", text)

    return {
        "min_degree_level": degree.group(1) if degree else None,
        "min_marks_general": float(gen.group(1)) if gen else None,
        "min_marks_reserved": float(res.group(1)) if res else None
    }

def main():
    students_df = pd.read_csv("recommender_system/students_data.csv")
    courses_df = pd.read_csv("recommender_system/final_data3.csv")
    eligibility_pipe = EligibilityRule()
    tfidf = TFIDFVector()

    results = []

    for (i, student) in students_df.iterrows():
        eligible_courses_text = []
        for (j, course) in courses_df.iterrows():
            try:
                result = eligibility_pipe.check_eligibility(student, course)

                if result[0]:
                    text = f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(course['skills_learned'])} {" ".join(course['career_outcomes'])}"""
                    eligible_courses_text.append(text)
                else:
                    continue
            except:
                print(f"Some problem with this course {course["program_name"]}")
                continue
        print(f"Student {i + 1} eligible for {len(eligible_courses_text)} courses.")
        student_text = f"""{student['academic_background']} {student['subjects']} {student['preferred_skills']} {student['preferred_domain']} {student['career_goal']} {student['interests']}"""
        tfidf.fit(eligible_courses_text)
        course_matrix = tfidf.transform(eligible_courses_text)
        student_vector = tfidf.transform(student_text)

        scores = calculate_similarity(student_vector=[student_vector], courses_matrix=course_matrix)
        
        print(course_matrix.shape)

        
        

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error occured: ", str(e))
        import traceback
        traceback.print_exc()