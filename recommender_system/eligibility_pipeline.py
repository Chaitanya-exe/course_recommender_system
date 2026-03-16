import pandas as pd

class EligibilityRule:

    def __init__(self):
        self.degree_order = {
            "PreUG": 0,
            "UG" : 1,
            "PG" : 2,
            "PhD" : 3
        }                            
    
    def _check_degree(self, student, course) -> bool:
        return self.degree_order[student.degree_level] >= self.degree_order[course["min_degree_level"]]
    
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
    
    def check_eligibility(self, student, course):
        if not self._check_degree(student, course):
            return False, "degree not eligible"
        if not self._check_marks(student, course):
            return False, "marks not eligible"
        
        return True, "Eligible"