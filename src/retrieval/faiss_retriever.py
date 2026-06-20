import json
import pickle

import faiss
import numpy as np

from pathlib import Path

from sentence_transformers import (
    SentenceTransformer
)


JOBS_PATH = Path(
    "data/processed/jobs.json"
)

INDEX_PATH = Path(
    "artifacts/indexes/faiss.index"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class FAISSRetriever:

    def __init__(self):

        with open(
            JOBS_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            self.jobs = json.load(f)

        self.model = (
            SentenceTransformer(
                EMBEDDING_MODEL
            )
        )

        self.index = (
            faiss.read_index(
                str(INDEX_PATH)
            )
        )

    def search(
        self,
        query,
        top_k=3
    ):

        query_embedding = (
            self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True

            )
            .astype("float32")
        )

        distances, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for score, idx in zip(
            distances[0],
            indices[0]
        ):

            results.append(
                {
                    "score": float(score),
                    "job": self.jobs[idx]
                }
            )

        return results