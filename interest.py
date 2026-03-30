import re
from groq import Groq
from pypdf import PdfReader

MODEL = "llama-3.1-8b-instant"

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()

def groq_chat(client, system, user, tokens=700):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.6,
        max_tokens=tokens
    )
    return response.choices[0].message.content.strip()

def resume_agent(client, text):
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
{text}
"""
    return groq_chat(client, "Resume Analyzer", prompt, 600)

def interest_agent(client, interests):
    prompt = f"""
User Interests: {interests}

Suggest EXACTLY 3 careers.

Format:
1. Career → short reason
2. Career → short reason
3. Career → short reason
"""
    return groq_chat(client, "Interest Analyzer", prompt, 400)

def final_agent(client, data):
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
{data}
"""
    return groq_chat(client, "Career Expert", prompt, 700)

def generate_linkedin_link(role):
    return "https://www.linkedin.com/jobs/search/?keywords=" + role.replace(" ", "%20") + "&location=India"

def extract_career(text):
    match = re.search(r"Career:\s*(.+)", text)
    return match.group(1).strip() if match else "Career"

def run_career_recommender(api_key, resume_path, interests=""):
    client = Groq(api_key=api_key)

    resume_text = extract_text_from_pdf(resume_path)

    resume_output = resume_agent(client, resume_text) if resume_text else ""
    interest_output = interest_agent(client, interests) if interests else ""

    combined_data = f"""
Resume:
{resume_output}

User Interests:
{interests}

Interest Analysis:
{interest_output}
"""

    final_output = final_agent(client, combined_data)

    career = extract_career(final_output)
    job_link = generate_linkedin_link(career)

    return {
        "resume_output": resume_output,
        "interest_output": interest_output,
        "final_output": final_output,
        "career": career,
        "job_link": job_link
    }
