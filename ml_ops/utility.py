import sqlite3
import pandas as pd
import numpy as np
import ast


def load_training_dataset():

    conn=sqlite3.connect(
        "experiment.db"
    )

    df=pd.read_sql(
        "SELECT * FROM dataset",
        conn
    )

    conn.close()

    json_cols=[

    "subjects",
    "interests",
    "preferred_skills",

    "skills_learned",

    "career_outcomes",

    "required_subjects"

    ]

    for col in json_cols:

        df[col] = df[col].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x))




    bool_cols=[

    "is_reserved",

    "domain_match",

    "subject_required",

    "marks_required"

    ]

    for col in bool_cols:

        df[col]=df[col].astype(bool)


    numeric_cols=[

    "percentage",

    "similarity_score",

    "subject_overlap",

    "marks_margin",

    "skill_relevance_percentage",

    "career_align_percentage",

    "label"

    ]

    for col in numeric_cols:

        df[col]=pd.to_numeric(
            df[col],
            errors="coerce"
        )
    df['min_marks_general'] = (
        df["min_marks_general"].fillna(-1)
    )
    df['min_marks_reserved'] = (
        df["min_marks_reserved"].fillna(-1)
    )
    df['marks_margin'] = np.where(
        df['marks_required'] == 0,
        -1,
        df['marks_margin']
    )
    df['duration'] = (
        df['duration'].fillna("N/A")
    )
    return df
