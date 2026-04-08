from datetime import datetime
import json
from .database import get_connection

def save_result(student, scores, courses, feedback, method):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO recommendation_logs
        (timestamp, student_profile, method, recommended_courses, scores, feedback)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        json.dumps(student),
        method,
        json.dumps([c['program_name'] for c in courses]),
        json.dumps(scores),
        feedback
    ))

    conn.commit()
    conn.close()