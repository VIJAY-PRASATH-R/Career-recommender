# CareerAI – Multi-Agent Career Recommender

A web app that analyzes your resume using **4 AI agents** to recommend the most suitable career path.

## How it works

1. Upload your resume (PDF)
2. Enter your Groq API key (free at [console.groq.com](https://console.groq.com))
3. Four agents collaborate and deliver a personalized career recommendation

| Agent | Role |
|-------|------|
| 🧠 Skill Agent | Extracts skills from your resume |
| 💡 Interest Agent | Suggests careers based on your interests |
| 📊 Market Agent | Scores market demand for each career |
| 💰 Salary Agent | Provides India + global salary ranges |

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000**

## Tech Stack

- **Backend:** Flask, Groq LLM, LangChain
- **PDF Parsing:** pypdf
- **Frontend:** Vanilla HTML/CSS/JS