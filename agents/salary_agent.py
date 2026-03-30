"""
Salary Agent
============
Provides salary range insights for a list of careers using Groq LLM.
This is a new agent that complements the existing skill/interest/market agents.
"""

import json
import re
from groq import Groq

MODEL = "llama-3.1-8b-instant"


def run_salary_agent(careers: list, groq_api_key: str) -> dict:
    """
    Generate salary range data for a list of careers.

    Args:
        careers:      List of career role names.
        groq_api_key: Groq API key.

    Returns:
        dict with key 'salaries': list of dicts, each containing:
            - career      : role name
            - india_range : salary range in India (LPA)
            - global_range: salary range globally (USD/year)
            - growth      : growth outlook string
    """
    client = Groq(api_key=groq_api_key)

    careers_str = ", ".join(careers)
    prompt = f"""
You are a compensation expert. Provide salary data for these careers: {careers_str}

Return ONLY a JSON object (no markdown, no extra text):
{{
  "salaries": [
    {{
      "career": "<role>",
      "india_range": "<e.g. ₹8–18 LPA>",
      "global_range": "<e.g. $80,000–$130,000/yr>",
      "growth": "<e.g. High – 25% growth expected in 3 years>"
    }}
  ]
}}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Compensation data expert. Return only JSON."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: return minimal structure so the UI never crashes
        return {
            "salaries": [
                {"career": c, "india_range": "N/A", "global_range": "N/A", "growth": "N/A"}
                for c in careers
            ]
        }
