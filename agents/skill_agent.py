"""
Skill Agent
===========
Extracts skills from a resume PDF using pypdf (text extraction) + Groq LLM.

Refactored from skill_analyzer.py:
  - Removed google.colab dependency (Colab-only)
  - Removed LlamaIndex/HuggingFace embeddings (caused tf-keras import conflict)
  - Uses the same Groq client pattern as all other agents for consistency
"""

import re
from groq import Groq
from pypdf import PdfReader

MODEL = "llama-3.1-8b-instant"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using pypdf."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


def run_skill_agent(pdf_path: str, groq_api_key: str) -> dict:
    """
    Extract skills from a resume PDF.

    Args:
        pdf_path:     Absolute path to the uploaded resume PDF.
        groq_api_key: Groq API key for the LLM.

    Returns:
        dict with keys:
            - skills     : list of extracted skill strings
            - raw_text   : full resume text (passed to downstream agents)
            - raw_output : raw LLM response string
    """
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text:
        return {"skills": [], "raw_text": "", "raw_output": "No text found in PDF."}

    client = Groq(api_key=groq_api_key)

    prompt = f"""
You are a resume parser. Extract ALL skills from the resume below.
Include: programming languages, frameworks, libraries, tools, cloud platforms,
databases, methodologies, and soft skills.

Return ONLY a comma-separated list of skills. No explanations, no numbering.

Resume:
{resume_text[:4000]}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Resume skill extractor. Return comma-separated skills only."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    raw_output = response.choices[0].message.content.strip()

    # Parse comma-separated list, remove empty strings and duplicates
    skills = list(dict.fromkeys(
        s.strip() for s in re.split(r"[,\n]", raw_output)
        if s.strip() and len(s.strip()) < 60
    ))

    return {
        "skills": skills,
        "raw_text": resume_text,
        "raw_output": raw_output,
    }
