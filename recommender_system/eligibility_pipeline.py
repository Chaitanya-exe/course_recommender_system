from pydantic import BaseModel, Field
from typing import List
from restructure_pipeline.dataset_classes import DatasetRow

class StudentProfile(BaseModel):
    degree_level: str = Field(description="student's current degree level")
    percentage: float = Field(description="student's latest percentage in the degree")
    is_reserved: bool = Field(desctiption="whether student belong to a reserved category or not")
    academic_background: str = Field(description="student's degree name and subject")
    subjects: List[str] = Field(description="subjects studied in the course")
    interests: List[str] = Field(description="student's area of interests")
    career_goal: str = Field(desctiption="student's career goal")
    preferred_skills: List[str] = Field(description="student's preferred skills to learn")
    preferred_domain: str = Field(description="student's preferred domain")


class EligibilityRule:

    def __init__(self):
        self.degree_order = {
            "preUG": 0,
            "UG" : 1,
            "PG" : 2,
            "PhD" : 3
        }                            
    
    def _check_degree(self, student: StudentProfile, course: DatasetRow) -> bool:
        return self.degree_order[student.degree_level] >= self.degree_order[course.eligibility_struct.min_degree_level]
    
    def _check_marks(self, student: StudentProfile, course: DatasetRow) -> bool:
        if student.is_reserved:
            return student.percentage >= course.eligibility_struct.min_marks_reserved
        
        return student.percentage >= course.eligibility_struct.min_marks_general
    
    def check_eligibility(self, student: StudentProfile, course: DatasetRow):
        if not self._check_degree(student, course):
            return False, "degree not eligible"
        if not self._check_marks(student, course):
            return False, "marks not eligible"
        
        return True, "Eligible"