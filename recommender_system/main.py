from eligibility_pipeline import EligibilityRule
import pandas as pd
import re

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

    results = []

    for (i, student) in students_df.iterrows():
        eligible_courses = []
        for (j, course) in courses_df.iterrows():
            try:
                result = eligibility_pipe.check_eligibility(student, course)

                if result[0]:
                    eligible_courses.append(course)
                else:
                    continue
            except:
                print(f"Some problem with this course {course["program_name"]}")
                continue
        
        print(f"Student {i + 1} eligible for {len(eligible_courses)} courses.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error occured: ", str(e))
        import traceback
        traceback.print_exc()