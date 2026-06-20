from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.baselines.tfidf import TFIDFRetriever
from src.baselines.bm25 import BM25Retriever

from src.retrieval.faiss_retriever import (
    FAISSRetriever
)

query = """I am looking for a role where I can develop intelligent software that learns from large amounts of data and improves its performance over time.

I enjoy solving complex analytical problems, building systems that can understand human language, and creating applications that assist users through automated conversations.

My experience includes designing predictive solutions, improving decision-making through data-driven approaches, and collaborating with teams to deploy innovative technologies into real-world products.

I am interested in positions that involve research, model development, and the creation of smart applications for businesses and consumers."""

def print_results(title, results):

    print("\n")
    print("=" * 80)

    print(title)

    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        job = result["job"]

        print("\n" + "=" * 80)

        print(f"Rank #{rank}")

        print(f"Score: {result['score']:.4f}")

        print(f"Job Title: {job['title']}")

        print(f"Experience Level: {job['experience_level']}")

        print(f"\nSkills:\n{job['skills']}")

        print(f"\nResponsibilities:\n{job['responsibilities']}")

        print(f"\nKeywords:\n{job['keywords']}")

        print("=" * 80)


print_results(
    "TF-IDF",
    TFIDFRetriever().search(
        query
    )
)

print_results(
    "BM25",
    BM25Retriever().search(
        query
    )
)

print_results(
    "FAISS",
    FAISSRetriever().search(
        query
    )
)