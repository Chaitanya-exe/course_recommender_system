from pydantic import BaseModel, Field
from typing import Optional, List

class EligibilityStruct(BaseModel):
    min_degree_level: Optional[str] = Field(
        default=None,
        description="One of: PreUG, UG, PG, PhD"
    )
    min_marks_general: Optional[float] = Field(
        default=None,
        description="Minimum percentage for general category",
        ge=0.0,
        le=100.0
    )
    min_marks_reserved: Optional[float] = Field(
        default=None,
        description="Minimum percentage for reserved category (SC/ST/OBC/PwD)",
        ge=0.0,
        le=100.0
    )

class DatasetRow(BaseModel):
    program_name: str = Field(description="Official name of the academic program")
    program_level: str = Field(description="UG, PG, Diploma, Certificate, or PhD level")
    domain: str = Field(description="Broad academic domain such as Engineering, Science, Management, Humanities, etc.")
    eligibility: str = Field(description="Eligibility criteria required for admission")
    description: str = Field(description="Short academic description of the program")
    skills_learned: List[str] = Field(
        description="Key academic or professional skills gained after completing the program"
    )
    career_outcomes: List[str] = Field(
        description="Typical career paths or job roles after completing the program"
    )
    eligibility_struct: EligibilityStruct = Field(description="Eligibiliy in the structured format")