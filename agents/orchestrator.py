"""
Orchestrator
============
Coordinates all four agents in sequence and returns a single unified result dict.

Pipeline:
  Resume PDF
      │
      ▼
  [Skill Agent]  ──→ skills list + raw_text
      │
      ▼
  [Interest Agent]  ──→ career suggestions + final recommendation + LinkedIn link
      │
      ▼
  [Market Agent]  ──→ demand scores for suggested careers
      │
      ▼
  [Salary Agent]  ──→ salary ranges
      │
      ▼
  Combined result dict  ──→  results.html
"""

import re
from agents.skill_agent import run_skill_agent
from agents.interest_agent import run_interest_agent
from agents.market_agent import run_market_demand_agent
from agents.salary_agent import run_salary_agent


def _extract_career_names_from_suggestions(suggestions_text: str) -> list:
    """
    Parse career names out of the numbered list produced by the interest agent.
    Expected format: "1. Career Name → reason"
    """
    careers = []
    for line in suggestions_text.splitlines():
        match = re.match(r"^\d+\.\s+([^→\-–]+)", line)
        if match:
            careers.append(match.group(1).strip())
    return careers if careers else ["Software Engineer", "Data Analyst", "Product Manager"]


def run_pipeline(pdf_path: str, groq_api_key: str, extra_interests: str = "") -> dict:
    """
    Run the full multi-agent pipeline.

    Args:
        pdf_path:        Absolute path to the uploaded resume PDF.
        groq_api_key:    Groq API key used by all agents.
        extra_interests: Optional interests added by the user via the form.

    Returns:
        A combined dict consumed by results.html:
        {
            "skills":          list[str],
            "resume_summary":  str,
            "interests":       str,
            "career_suggestions": str,
            "final_output":    str,
            "career":          str,
            "job_link":        str,
            "market":          dict  (from market agent),
            "salaries":        list  (from salary agent),
            "error":           str | None
        }
    """
    result: dict = {
        "skills": [],
        "resume_summary": "",
        "interests": "",
        "career_suggestions": "",
        "final_output": "",
        "career": "",
        "job_link": "",
        "market": {},
        "salaries": [],
        "error": None,
    }

    try:
        # ── Agent 1 : Skill Extraction ─────────────────────────────────────
        skill_data = run_skill_agent(pdf_path, groq_api_key)
        result["skills"] = skill_data["skills"]
        resume_text = skill_data["raw_text"]

        # ── Agent 2 : Interest & Final Recommendation ──────────────────────
        interest_data = run_interest_agent(resume_text, groq_api_key, extra_interests)
        result["resume_summary"]    = interest_data["resume_summary"]
        result["career_suggestions"]= interest_data["career_suggestions"]
        result["final_output"]      = interest_data["final_output"]
        result["career"]            = interest_data["career"]
        result["job_link"]          = interest_data["job_link"]
        result["interests"]         = interest_data["interests"]

        # ── Agent 3 : Market Demand ────────────────────────────────────────
        career_list = _extract_career_names_from_suggestions(
            interest_data["career_suggestions"]
        )
        # Add the final recommended career if not already in the list
        if result["career"] and result["career"] not in career_list:
            career_list.insert(0, result["career"])

        market_data = run_market_demand_agent(
            careers=career_list,
            skills=result["skills"][:5],   # top 5 skills for context
            groq_api_key=groq_api_key,
        )
        result["market"] = market_data

        # ── Agent 4 : Salary Insights ──────────────────────────────────────
        salary_data = run_salary_agent(career_list, groq_api_key)
        result["salaries"] = salary_data.get("salaries", [])

    except Exception as exc:
        result["error"] = str(exc)

    return result
