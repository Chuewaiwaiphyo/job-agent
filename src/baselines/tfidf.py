import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/processed/jobs.json")


class TFIDFRetriever:

    def __init__(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.jobs = json.load(f)

        self.documents = [
            job["document"]
            for job in self.jobs
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,2),
            max_features=10000
)

        self.doc_vectors = self.vectorizer.fit_transform(
            self.documents
        )

    def search(
        self,
        query: str,
        top_k: int = 3
    ):
        query_vector = self.vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            self.doc_vectors
        )[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for idx in ranked_indices:
            results.append(
                {
                    "score": float(scores[idx]),
                    "job": self.jobs[idx]
                }
            )

        return results