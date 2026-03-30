"""
Flask Application – Multi-Agent Career Recommender
===================================================
Entry point for the web application.

Routes:
    GET  /          → index.html (resume upload form)
    POST /analyze   → runs all agents → results.html
    GET  /health    → status check
"""

import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash

from agents.orchestrator import run_pipeline

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "career-recommender-secret-2025")

# Max upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── Validate inputs ────────────────────────────────────────────────────
    groq_key = request.form.get("groq_key", "").strip()
    if not groq_key:
        flash("Please enter your Groq API key.", "error")
        return redirect(url_for("index"))

    if "resume" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("index"))

    file = request.files["resume"]
    if file.filename == "" or not _allowed_file(file.filename):
        flash("Please upload a valid PDF resume.", "error")
        return redirect(url_for("index"))

    extra_interests = request.form.get("interests", "").strip()

    # ── Save PDF to temp file ───────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── Run multi-agent pipeline ────────────────────────────────────────
        result = run_pipeline(
            pdf_path=tmp_path,
            groq_api_key=groq_key,
            extra_interests=extra_interests,
        )
    finally:
        # Always clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.get("error"):
        flash(f"Analysis error: {result['error']}", "error")
        return redirect(url_for("index"))

    return render_template("results.html", result=result)


@app.route("/health")
def health():
    return {"status": "ok", "service": "Career Recommender"}, 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
