"""Input filter: classifies patient messages before they reach the LLM."""
import json
import re
import ollama

_INPUT_CHECKER_PROMPT = """\
You are screening a patient's message before it reaches a CBT therapist.

Classify the message as:
- PASS: genuine therapeutic engagement — sharing feelings, thoughts, experiences, asking questions,
  expressing resistance, frustration, or confusion. Even short emotional responses ("I don't know",
  "I feel bad", "not great") are PASS. Replies in any language are PASS.
- REDIRECT: no therapeutic value — system tests ("again", "do it again", "repeat"), gibberish,
  purely technical commands, or messages clearly intended to probe the system rather than engage.

Reply with JSON only — no text outside the JSON:
{"verdict": "PASS"}
or
{"verdict": "REDIRECT", "reply": "..."}

If REDIRECT, reply must be a single warm sentence in the same language as the patient's message,
acknowledging them and gently inviting genuine engagement.\
"""


def check_input(patient_message: str, model: str) -> dict:
    """
    Returns {"verdict": "PASS"} or {"verdict": "REDIRECT", "reply": str}.
    On any error, returns PASS so the checker never silently blocks delivery.
    """
    messages = [
        {
            "role": "user",
            "content": f"{_INPUT_CHECKER_PROMPT}\n\nPatient message:\n{patient_message}",
        }
    ]
    try:
        text = ""
        for chunk in ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0},
            stream=True,
        ):
            text += chunk["message"]["content"]
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"verdict": "PASS"}
    except Exception as exc:
        return {"verdict": "PASS", "reason": f"Checker error (skipped): {exc}"}
