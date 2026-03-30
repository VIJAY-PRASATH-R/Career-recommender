"""
Interest Agent
==============
Analyses resume text to extract interests and suggest career paths.
Adapted directly from interest.py.
"""

import re
from groq import Groq

MODEL = "llama-3.1-8b-instant"


def _groq_chat(client: Groq, system: str, user: str, tokens: int = 700) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.6,
        max_tokens=tokens,
    )
    return response.choices[0].message.content.strip()


def _resume_agent(client: Groq, resume_text: str) -> str:
    prompt = f"""
Extract ONLY TOP 3 most important:
- Skills
- Interests
Also give:
- Domain

Keep output SHORT.

Format:
Skills: skill1, skill2, skill3
Interests: interest1, interest2, interest3
Domain: one domain

Resume:
{resume_text}
"""
    return _groq_chat(client, "Resume Analyzer", prompt, 600)


def _interest_career_agent(client: Groq, interests: str) -> str:
    prompt = f"""
User Interests: {interests}

Suggest EXACTLY 3 careers.

Format:
1. Career → short reason
2. Career → short reason
3. Career → short reason
"""
    return _groq_chat(client, "Interest Analyzer", prompt, 400)


def _final_recommendation_agent(client: Groq, combined_data: str) -> str:
    prompt = f"""
Based on user data, suggest ONE best career.

Format:
🎯 Career: <role>

💡 Why this fits:
<2-3 lines>

💰 Salary (India):
<range>

📈 Demand:
<short line>

🛣️ Roadmap:
1. step
2. step
3. step

User Data:
{combined_data}
"""
    return _groq_chat(client, "Career Expert", prompt, 700)


def _generate_linkedin_link(role: str) -> str:
    return (
        "https://www.linkedin.com/jobs/search/?keywords="
        + role.replace(" ", "%20")
        + "&location=India"
    )


def _extract_career_name(text: str) -> str:
    match = re.search(r"Career:\s*(.+)", text)
    return match.group(1).strip() if match else "Career"


def run_interest_agent(resume_text: str, groq_api_key: str, extra_interests: str = "") -> dict:
    """
    Run the interest/career recommendation pipeline.

    Args:
        resume_text:     Full text extracted from the resume.
        groq_api_key:    Groq API key.
        extra_interests: Optional additional interests typed by the user.

    Returns:
        dict with keys:
            - resume_summary    : extracted skills/interests/domain
            - career_suggestions: 3 interest-based career suggestions
            - final_output      : full final recommendation text
            - career            : extracted career role name
            - job_link          : LinkedIn search URL
            - interests         : extracted interests string
    """
    client = Groq(api_key=groq_api_key)

    resume_summary = _resume_agent(client, resume_text)

    # Extract interests section from summary for follow-up
    interests_match = re.search(r"Interests:\s*(.+)", resume_summary)
    extracted_interests = interests_match.group(1).strip() if interests_match else ""
    all_interests = ", ".join(
        filter(None, [extracted_interests, extra_interests])
    )

    career_suggestions = _interest_career_agent(client, all_interests) if all_interests else ""

    combined_data = f"""
Resume Summary:
{resume_summary}

User Interests:
{all_interests}

Career Suggestions:
{career_suggestions}
"""
    final_output = _final_recommendation_agent(client, combined_data)

    career = _extract_career_name(final_output)
    job_link = _generate_linkedin_link(career)

    return {
        "resume_summary": resume_summary,
        "career_suggestions": career_suggestions,
        "final_output": final_output,
        "career": career,
        "job_link": job_link,
        "interests": all_interests,
    }
