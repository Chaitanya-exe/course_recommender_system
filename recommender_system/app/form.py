from eligibility_pipeline import EligibilityRule
from utils.utility import get_string_from_array, Finaliser
from vector_engine import TFIDFEngine, EmbeddingEngine, HybridEngine
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_csv("recommender_system/final_data3.csv")

courses_df = load_data()

@st.cache_resource
def init_engines(corpus):
    tfidf_engine = TFIDFEngine()
    embed_engine = EmbeddingEngine()
    hybrid_engine = HybridEngine(tfidf=tfidf_engine, embedder=embed_engine)
    tfidf_engine.fit(corpus)
    return tfidf_engine, embed_engine, hybrid_engine

courses_text = [f"""{course['program_name']} {course['domain']} {course['description']} {" ".join(get_string_from_array(course['skills_learned']))} {" ".join(get_string_from_array(course['career_outcomes']))}""" for (i, course) in courses_df.iterrows()]
tfidf, embed, hybrid = init_engines(courses_text)

