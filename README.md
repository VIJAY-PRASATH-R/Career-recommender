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

## Setup & Deployment

### 🌐 Vercel (Web App)
The `requirements.txt` is optimized for Vercel's 250MB limit. Simply import this repo into Vercel to deploy.

### 💻 Local Development (Full)
To run everything locally, including the original legacy scripts:
```bash
pip install -r requirements-dev.txt
python app.py
```

## Project Structure
- `app.py`: Main Flask web app.
- `agents/`: The multi-agent logic.
- `skill_analyzer.py`, `interest.py`, `market_agent.py`: Original legacy scripts.

## Tech Stack
- **Backend:** Flask, Groq LLM, LangChain
- **PDF Parsing:** pypdf
- **Frontend:** Vanilla HTML/CSS/JS (Minimalist Design)