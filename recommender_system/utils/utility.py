import ast
import pandas as pd

def get_string_from_array(input: str) -> list[any]:
    return ast.literal_eval(input)

class Finaliser:
    def __init__(self):
        self.records = []

    def add_record(
            self,
            student_id,
            recommendations,
            algorithm,
            student_domain,
            scores
    ):
        for rank, (course, score) in enumerate(zip(recommendations, scores), start=1):
            self.records.append({
                "student_id": student_id,
                "algorithm": algorithm,
                "rank": rank,
                "course_name": course['program_name'],
                "course_domain": course['domain'],
                "score": score,
                "student_domain": student_domain
            })
    
    def to_csv(self, filename = "results.csv"):
        pd.DataFrame(self.records).to_csv(filename, index=False)
