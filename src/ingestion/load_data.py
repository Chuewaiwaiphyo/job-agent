from pathlib import Path
import json
import pandas as pd


RAW_DATA_PATH = Path("data/raw/job_dataset.csv")
PROCESSED_DATA_PATH = Path("data/processed/jobs.json")


def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = " ".join(text.split())

    return text


def load_dataset():

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"\nOriginal rows: {len(df)}")

    # Remove duplicate jobs
    df = df.drop_duplicates()

    # Remove rows with missing titles
    df = df.dropna(subset=["Title"])

    print(f"Rows after cleaning: {len(df)}")

    return df


def build_document(row):

    title = clean_text(row["Title"])
    level = clean_text(row["ExperienceLevel"])
    years = clean_text(row["YearsOfExperience"])
    skills = clean_text(row["Skills"])
    responsibilities = clean_text(row["Responsibilities"])
    keywords = clean_text(row["Keywords"])

    document = f"""
Job Title: {title}

Experience Level: {level}

Years of Experience: {years}

Skills:
{skills}

Responsibilities:
{responsibilities}

Keywords:
{keywords}
""".strip()

    return {
        "job_id": str(row["JobID"]),
        "title": title,
        "experience_level": level,
        "years_of_experience": years,
        "skills": skills,
        "responsibilities": responsibilities,
        "keywords": keywords,
        "document": document
    }


def process_dataset(df):

    jobs = []

    for _, row in df.iterrows():
        jobs.append(
            build_document(row)
        )

    return jobs


def save_jobs(jobs):

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        PROCESSED_DATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            jobs,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved {len(jobs)} jobs")


def main():

    df = load_dataset()

    jobs = process_dataset(df)

    save_jobs(jobs)


if __name__ == "__main__":
    main()