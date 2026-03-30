"""
Market Demand Analyzer Agent
============================
A modular LangChain-based agent that analyzes job market demand for a list of careers.
Designed to plug cleanly into a larger multi-agent system.

Dependencies:
    pip install langchain langchain-groq
"""

import json
import re
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "llama-3.1-8b-instant"  # Groq's latest LLM optimized for reasoning and structured output
MAX_TOKENS = 2048
TEMPERATURE = 0.3  # Lower = more deterministic/consistent scoring


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

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


def build_prompt(
    careers: list[str],
    skills: Optional[list[str]] = None,
    news_trends: Optional[str] = None,
) -> str:
    """
    Build the formatted prompt string for the LLM.

    Args:
        careers:     List of career titles to analyze.
        skills:      Optional list of user skills to personalize scoring.
        news_trends: Optional string with recent news or market signals.

    Returns:
        Fully rendered prompt string.
    """
    careers_str = ", ".join(careers)

    skills_section = (
        f"User's current skills (factor these into demand alignment): {', '.join(skills)}"
        if skills
        else ""
    )

    news_section = (
        f"Recent news/market signals to incorporate:\n{news_trends}"
        if news_trends
        else ""
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


# ---------------------------------------------------------------------------
# Output Parser
# ---------------------------------------------------------------------------

def parse_output(raw_output: str) -> dict:
    """
    Parse the LLM's raw string output into a Python dictionary.

    Handles edge cases where the model wraps JSON in markdown fences.

    Args:
        raw_output: Raw string returned by the LLM.

    Returns:
        Parsed dictionary matching the expected schema.

    Raises:
        ValueError: If valid JSON cannot be extracted.
    """
    # Strip markdown code fences if present (defensive)
    cleaned = re.sub(r"```(?:json)?", "", raw_output).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse LLM output as JSON.\n"
            f"Raw output:\n{raw_output}\n"
            f"Error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------

def run_market_demand_agent(
    careers: list[str],
    skills: Optional[list[str]] = None,
    news_trends: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> dict:
    """
    Run the Market Demand Analyzer Agent.

    Args:
        careers:          List of career titles to evaluate.
        skills:           Optional user skill set for personalized analysis.
        news_trends:      Optional recent news string to enrich context.
        groq_api_key: Optional API key (falls back to GROQ_API_KEY env var).

    Returns:
        Structured dictionary with per-career analysis, top recommendation,
        and overall market insight.

    Raises:
        ValueError: On empty careers list or JSON parse failure.
        Exception:  On LLM API errors.
    """
    # --- Input validation ---
    if not careers:
        raise ValueError("The 'careers' list must contain at least one career.")

    # --- Build prompt ---
    prompt_str = build_prompt(careers, skills, news_trends)

    # --- Initialize LLM ---
    llm_kwargs = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if anthropic_api_key:
        llm_kwargs["api_key"] = anthropic_api_key

    llm = ChatGroq(**llm_kwargs)

    # --- Build chain: prompt → LLM → raw string ---
    chain = llm | StrOutputParser()

    # --- Invoke chain ---
    raw_output = chain.invoke(prompt_str)

    # --- Parse and return structured result ---
    return parse_output(raw_output)


# ---------------------------------------------------------------------------
# Pretty Printer (for CLI / demo use)
# ---------------------------------------------------------------------------

def print_analysis(result: dict) -> None:
    """
    Print the analysis result in a human-readable format.

    Args:
        result: Parsed dictionary from run_market_demand_agent().
    """
    print("\n" + "=" * 60)
    print("  MARKET DEMAND ANALYSIS REPORT")
    print("=" * 60)

    for item in result.get("analysis", []):
        score_bar = "█" * item["demand_score"] + "░" * (10 - item["demand_score"])
        print(f"\n  Career       : {item['career']}")
        print(f"  Demand Score : [{score_bar}] {item['demand_score']}/10")
        print(f"  Confidence   : {item['confidence']}")
        print(f"  Reason       : {item['reason']}")
        print("  " + "-" * 56)

    print(f"\n  ⭐ TOP RECOMMENDATION : {result.get('top_recommended_career', 'N/A')}")
    print(f"\n  📊 OVERALL INSIGHT\n  {result.get('overall_insight', '')}")
    print("\n" + "=" * 60 + "\n")
