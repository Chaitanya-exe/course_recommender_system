from vector_engine import TFIDFEngine, EmbeddingEngine, HybridEngine
from utils.utility import get_string_from_array
import pandas as pd

class EligibilityRule:

    def __init__(self, vectoriser):
        self.degree_order = {
            "PreUG": 0,
            "UG" : 1,
            "PG" : 2,
            "PhD" : 3
        }  
        self.vectoriser = vectoriser
    
    def _check_degree(self, student, course) -> bool:
        return self.degree_order[student.degree_level] == self.degree_order[course["min_degree_level"]]
    
    def _check_marks(self, student, course) -> bool:

        reserved_marks = course["min_marks_reserved"]
        general_marks = course["min_marks_general"]

        if student.is_reserved:
            if pd.isna(reserved_marks):
                return True
            return student.percentage >= reserved_marks

        if pd.isna(general_marks):
            return True
        return student.percentage >= general_marks
    
    def _check_domain(self, student, course):

        student_text = f"""{student['academic_background']} {" ".join(get_string_from_array(student['subjects']))} """.lower()
        course_text = f"{course['eligibility_keywords']}"

        student_vector = self.vectoriser.transform([student_text])
        course_vector = self.vectoriser.transform([course_text])

        if self.vectoriser.similarity(student_vector, course_vector) > 0.20:
            return True
        else:
            return False
        
    def check_eligibility(self, student, course):
        if not self._check_degree(student, course):
            return False, "degree not eligible"
        if not self._check_marks(student, course):
            return False, "marks not eligible"
        if not self._check_domain(student, course):
            return False, "subjects not eligible"
        
        return True, "Eligible"