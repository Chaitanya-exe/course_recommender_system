import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFVector:
    def __init__(self):
        self.base = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,2),
            max_features=5000
        )
        self.vectorizer = None

    def fit(self, corpus: list[str]):
        self.vectorizer = self.base.fit(corpus)

    def transform(self, texts: list[str]):
        if self.vectorizer is None:
            raise Exception("Vectorizer not fitted yet")

        return self.vectorizer.transform(texts)

class EmbeddingVector:
    def __init__(self):
        self.model_url = "http://127.0.0.1:11434/api/embeddings"
        self.model_name = "nomic-embed-text"
    
    def generate_embeddings(self, entity) -> list[float]:
        response = requests.post(url=self.model_url ,json={
            "model":self.model_name,
            "prompt": entity
        })

        if response.status_code != 200:
            raise Exception(f"Embedding API failed: {response.text}")

        data = response.json()

        return data["embedding"]
    
def calculate_similarity(student_vector, courses_matrix) -> list[float]:
    return cosine_similarity(student_vector, courses_matrix).flatten()