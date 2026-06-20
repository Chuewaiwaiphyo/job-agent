def build_resume_query(resume: dict):

    skills = resume.get(
        "skills",
        []
    )

    experience = resume.get(
        "experience",
        ""
    )

    projects = resume.get(
        "projects",
        []
    )

    query = f"""
Experience:
{experience}

Skills:
{' '.join(skills)}

Projects:
{' '.join(projects)}
"""

    return query.strip()