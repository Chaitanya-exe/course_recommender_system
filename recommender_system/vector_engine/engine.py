from .interface import VectorEngine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np

class TFIDFEngine(VectorEngine):

    def __init__(self):
        self.vectoriser = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,2),
            max_features=5000
        )

    def fit(self, corpus):
        self.vectoriser.fit(corpus)
    
    def transform(self, texts):
        return self.vectoriser.transform(texts)
    
    def similarity(self, query_vec, matrix):
        return cosine_similarity(query_vec, matrix).flatten()
    

class EmbeddingEngine(VectorEngine):
    
    def __init__(self):
        self.model_name = "nomic-embed-text"
        self.model_url = "http://127.0.0.1:11434/api/embeddings"
        self.matrix = None

    def generate_embeddings(self, entity) -> list[float]:
        response = requests.post(url=self.model_url ,json={
            "model":self.model_name,
            "prompt": entity
        })

        if response.status_code != 200:
            raise Exception(f"Embedding API failed: {response.text}")

        data = response.json()

        return data["embedding"]


    def fit(self, corpus):
        self.matrix = np.array([
            self.generate_embeddings(text)
            for text in corpus
        ])
    
    def transform(self, texts):
        return np.array([
            self.generate_embeddings(text)
            for text in texts
        ])
    
    def similarity(self, query_vec, matrix=None):

        matrix = matrix if matrix is not None else self.matrix
        dot = np.dot(matrix, query_vec.T).squeeze()
        norm = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        return dot / norm
    


class HybridEngine(VectorEngine):
    
    def __init__(self, tfidf: TFIDFEngine, embedder: EmbeddingEngine, alpha=0.7):
        self.tfidf = tfidf
        self.embedder = embedder
        self.alpha = alpha

    def fit(self, corpus):
        self.tfidf.fit(corpus)
        self.embedder.fit(corpus)
    
    def transform(self, texts):
        return {
            "tfidf": self.tfidf.transform(texts),
            "embeddings": self.embedder.transform(texts)
        }
    
    def similarity(self, query_vec, matrix):
        tfidf_score = self.tfidf.similarity(query_vec['tfidf'], matrix['tfidf'])
        embedding_score = self.embedder.similarity(query_vec['embeddings'], matrix['embeddings'])

        return self.alpha * embedding_score + (1 - self.alpha) * tfidf_score

