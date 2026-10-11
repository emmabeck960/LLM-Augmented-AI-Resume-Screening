import subprocess

def explain_resume(resume_text, decision, probability):
    status = "shortlisted" if decision == 1 else "rejected"

    prompt = f"""
You are an AI resume screening assistant.

Resume:
{resume_text}

Decision: {status}
Confidence score: {probability:.2f}

Explain in 3 short bullet points why the resume was {status}.
Be professional and constructive.
"""

    result = subprocess.run(
        ["ollama", "run", "phi3:mini"],
        input=prompt,
        text=True,
        capture_output=True
    )

    return result.stdout.strip()
