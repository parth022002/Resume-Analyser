"""
Curated RAG Knowledge Corpus for TalentForge AI Career Intelligence Platform.
Contains domain knowledge on ATS parsing guidelines, resume optimization benchmarks,
and interview readiness standards.
"""

CURATED_RAG_CORPUS = [
    {
        "id": "ats_rule_01",
        "category": "ats_guidelines",
        "title": "Single-Column Standard Layout",
        "content": "ATS screeners parse text top-to-bottom, left-to-right. Multi-column layouts, sidebars, tables, and text boxes often scramble text or get completely skipped by legacy ATS parsers like Workday and Taleo."
    },
    {
        "id": "ats_rule_02",
        "category": "ats_guidelines",
        "title": "Exact Skill Keyword Matching",
        "content": "ATS search algorithms match exact tokens and close synonyms. If a job description asks for 'React.js' and 'TypeScript', both full terms and abbreviations should appear naturally in experience bullet points."
    },
    {
        "id": "resume_opt_01",
        "category": "resume_benchmarks",
        "title": "Quantified Achievement Bullets (XYZ Formula)",
        "content": "High-scoring resumes follow Google's XYZ formula: Accomplished [X] as measured by [Y], by doing [Z]. Example: 'Decreased API response latency by 42% by introducing redis caching and async worker pools'."
    },
    {
        "id": "resume_opt_02",
        "category": "resume_benchmarks",
        "title": "Executive vs Technical Tone Calibration",
        "content": "Technical resumes emphasize tools, frameworks, and architecture patterns. Executive resumes emphasize ROI, cross-functional leadership, revenue impact, and team scaling metrics."
    },
    {
        "id": "interview_prep_01",
        "category": "interview_standards",
        "title": "STAR Method Behavioral Responses",
        "content": "Structure behavioral answers using Situation, Task, Action, and Result (STAR). Ensure 60% of the answer focuses on your specific actions and measurable outcomes."
    }
]
