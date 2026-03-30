"""
Market Demand Agent
===================
Scores market demand for a list of career paths using LangChain + Groq.
Adapted directly from market_agent.py.
"""

import json
import re
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL_NAME = "llama-3.1-8b-instant"
MAX_TOKENS = 2048
TEMPERATURE = 0.3

PROMPT_TEMPLATE = """
You are an expert labor market analyst with deep knowledge of global hiring trends,
technology adoption, and industry growth trajectories.

Analyze the job market demand for the following careers:
{careers}

Consider these factors for each career:
1. Current demand (job postings, hiring volume)
2. Future growth potential (3–5 year outlook)
3. Industry relevance (applicability across sectors)
4. Job availability (remote, hybrid, on-site options)
{skills_section}
{news_section}

Return your analysis STRICTLY as a JSON object with NO additional commentary or markdown.
The JSON must follow this exact structure:

{{
  "analysis": [
    {{
      "career": "<career name>",
      "demand_score": <integer 1–10>,
      "confidence": "<High|Medium|Low>",
      "reason": "<2–3 sentence explanation>"
    }}
  ],
  "top_recommended_career": "<career with highest demand_score>",
  "overall_insight": "<3–4 sentence market summary>"
}}

Rules:
- demand_score must be an integer between 1 and 10.
- confidence reflects how reliable your estimate is given available signals.
- reason must be concise but specific — cite real-world signals where possible.
- Return ONLY the JSON object. No markdown fences, no preamble, no trailing text.
"""


def _build_prompt(
    careers: list,
    skills: Optional[list] = None,
    news_trends: Optional[str] = None,
) -> str:
    careers_str = ", ".join(careers)
    skills_section = (
        f"User's current skills (factor these into demand alignment): {', '.join(skills)}"
        if skills else ""
    )
    news_section = (
        f"Recent news/market signals to incorporate:\n{news_trends}"
        if news_trends else ""
    )
    template = PromptTemplate(
        input_variables=["careers", "skills_section", "news_section"],
        template=PROMPT_TEMPLATE,
    )
    return template.format(
        careers=careers_str,
        skills_section=skills_section,
        news_section=news_section,
    )


def _parse_output(raw_output: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw_output).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse LLM output as JSON.\nRaw output:\n{raw_output}\nError: {exc}"
        ) from exc


def run_market_demand_agent(
    careers: list,
    skills: Optional[list] = None,
    news_trends: Optional[str] = None,
    groq_api_key: Optional[str] = None,
) -> dict:
    """
    Run the Market Demand Analyzer Agent.

    Args:
        careers:      List of career titles to evaluate.
        skills:       Optional user skill set for personalized analysis.
        news_trends:  Optional recent news string to enrich context.
        groq_api_key: Groq API key.

    Returns:
        Structured dict with per-career analysis, top recommendation,
        and overall market insight.
    """
    if not careers:
        raise ValueError("The 'careers' list must contain at least one career.")

    prompt_str = _build_prompt(careers, skills, news_trends)

    llm_kwargs = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if groq_api_key:
        llm_kwargs["api_key"] = groq_api_key

    llm = ChatGroq(**llm_kwargs)
    chain = llm | StrOutputParser()
    raw_output = chain.invoke(prompt_str)
    return _parse_output(raw_output)
