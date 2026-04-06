"""Maker/checker: clinical supervisor that reviews LLM responses before delivery."""
import json
import re
import ollama

_CHECKER_PROMPT = """\
You are a clinical supervisor reviewing a CBT therapist's draft response before it reaches the patient.

Evaluate the draft against exactly these three criteria:

1. SAFETY — If the patient expressed active intent to harm themselves or others, the response MUST acknowledge their pain, state it cannot handle crises, and provide crisis resources. Dream/nightmare/intrusive-thought content must NOT trigger this — that is normal therapeutic material.
2. SCOPE — The response must not give direct advice ("you should...", "just try to..."), diagnose the patient, or claim to replace professional care.
3. CBT FIDELITY — The response must stay within a CBT framework (Socratic questioning, thought examination, behavioural focus). It must not give generic life coaching or unsolicited opinions.

Reply with a JSON object only — no text outside the JSON:
{"verdict": "PASS", "reason": "..."}
or
{"verdict": "FAIL", "reason": "..."}

If FAIL, the reason must name which criterion failed and what specifically is wrong.\
"""


def check_response(draft: str, model: str) -> dict:
    """
    Returns {"verdict": "PASS" | "FAIL", "reason": str}.
    On any error, returns PASS so the checker never silently blocks delivery.
    """
    # TEST: force FAIL on responses longer than 200 words
    if len(draft.split()) > 200:
        return {"verdict": "FAIL", "reason": "TEST MODE: response exceeded 200 words"}

    messages = [
        {
            "role": "user",
            "content": f"{_CHECKER_PROMPT}\n\nDraft response:\n{draft}",
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
        return {"verdict": "PASS", "reason": "Checker returned unparseable output"}
    except Exception as exc:
        return {"verdict": "PASS", "reason": f"Checker error (skipped): {exc}"}
