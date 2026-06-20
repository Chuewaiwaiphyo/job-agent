from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi


DATA_PATH = Path("data/processed/jobs.json")


def tokenize(text: str):
    """
    BM25 tokenizer.
    Converts text to lowercase and removes punctuation.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


class BM25Retriever:

    def __init__(self):

        with open(
            DATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            self.jobs = json.load(f)

        self.documents = [
            job["document"]
            for job in self.jobs
        ]

        self.tokenized_docs = [
            tokenize(doc)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

    def search(
        self,
        query: str,
        top_k: int = 3
    ):

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = (
            scores.argsort()[::-1][:top_k]
        )

        results = []

        for idx in ranked_indices:

            results.append(
                {
                    "score": float(scores[idx]),
                    "job": self.jobs[idx]
                }
            )

        return results
