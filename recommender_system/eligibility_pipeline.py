from pydantic import BaseModel, Field
from typing import List

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
    
    def is_degree_matching(self, student_profile):
        pass