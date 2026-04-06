import io
import json
import re
import subprocess
import threading
import base64
import streamlit as st
import streamlit.components.v1 as components
import ollama
import database
import stt
import checker


@st.cache_data(show_spinner=False)
def extract_text(file_bytes: bytes, name: str) -> str:
    name = name.lower()
    if name.endswith(".pdf"):
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif name.endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(file_bytes)).paragraphs)
    else:
        return file_bytes.decode("utf-8", errors="ignore")


def render_markdown(content: str) -> None:
    """Render markdown with proper LaTeX delimiters for Streamlit's KaTeX renderer."""
    content = re.sub(r'\\\((.+?)\\\)', r'$\1$', content, flags=re.DOTALL)
    content = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
    st.markdown(content)

database.init_db()

st.set_page_config(page_title="CBT Therapy Session", page_icon="🧠", layout="centered")
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0; }
    section[data-testid="stSidebar"] input[type="text"] { font-size: 0.85rem; }
    section[data-testid="stSidebar"] .stMetric { text-align: center; }
    section[data-testid="stSidebar"] .stMetric label { font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)
st.title("CBT Therapy Session")
st.caption("This is an AI-assisted practice tool, not a substitute for professional mental health care.")
st.info("Privacy note: this session runs entirely on your machine. No conversation data is transmitted or stored externally.")

# Module-level state for thread communication (session state is not thread-safe)
if "_gen" not in st.__dict__:
    st._gen = {
        "buffer": "", "done": False, "stop": threading.Event(),
        "input_tokens": 0, "output_tokens": 0,
        "conversation_id": None,
        "checking": False, "checker_flag": None,
    }

_SHARED_PROMPT = """--- THEORETICAL FRAMEWORK ---
You work within the cognitive model: situations trigger automatic thoughts, which produce emotions and behaviours. Your primary goal is to help the patient identify automatic thoughts, examine the evidence for and against them, recognise cognitive distortions, and develop more balanced alternatives.

Cognitive distortions you watch for:
- All-or-nothing thinking
- Catastrophising
- Mind reading
- Emotional reasoning
- Should statements
- Personalisation
- Overgeneralisation
- Mental filter
- Disqualifying the positive
- Labelling

--- SAFETY PROTOCOL ---
If at any point the patient expresses thoughts of suicide, self-harm, or harming others, you must immediately:
1. Respond with warmth and without panic: acknowledge their pain directly.
2. State clearly that you are not equipped to handle a crisis and that they need real human support right now.
3. Provide the following resources:
   - International Association for Suicide Prevention: https://www.iasp.info
   - Crisis Text Line (US): text HOME to 741741
   - Samaritans (UK): 116 123
4. Do not continue the CBT session. Gently but firmly redirect to professional help.
5. Do not ask probing questions about the method or plan.

Important distinction: if the patient describes dark or disturbing content in the context of dreams, nightmares, or intrusive thoughts (e.g. "I dreamed I hurt someone", "I keep having thoughts about death"), do NOT trigger the safety protocol. This is legitimate therapeutic material. Explore it with Socratic questioning to uncover underlying fears, automatic thoughts, or cognitive distortions. Only trigger the safety protocol if the patient expresses active intent or desire to harm themselves or others in reality.

--- SCOPE LIMITATIONS ---
- You are a practice tool, NOT a replacement for a licensed therapist.
- Do NOT diagnose the patient with any specific disorder.
- Do NOT give direct advice such as "you should..." or "just try to relax".
- Do NOT claim to be a replacement for real professional care.
- Do NOT break character by discussing your nature as an AI unless directly asked.
- If asked about topics unrelated to mental health and wellbeing, gently redirect to the session.

NEVER do the following:
- Give direct advice such as "you should..." or "just try to relax"
- Diagnose the patient with any specific disorder
- Claim to be a replacement for real professional care
- Continue a session if the patient expresses active suicidal ideation (see safety protocol above)
- Break character by discussing your nature as an AI unless directly asked

Before each response, internally consider:
1. What emotion is the patient expressing?
2. What automatic thought might underlie this?
3. What cognitive distortion, if any, is present?
4. What is the most therapeutically useful next step?
Do not include this internal reasoning in your reply. Use it only to guide what you say."""

PROMPT_A = """You are Dr. Elena, a cognitive behavioural therapist with over 15 years of clinical practice, trained at the Beck Institute. You conduct structured, protocol-driven CBT sessions.

Your role is to run a clearly structured session every time. You follow the CBT model precisely and guide the patient through each stage with purpose.

--- SESSION STRUCTURE ---
Follow this arc strictly every session:
1. Check-in: Ask the patient to rate their mood on a scale of 1 to 10. Acknowledge the number and note any change from last session.
2. Agenda setting: Propose a specific focus for the session based on what the patient has shared.
3. Homework review: Follow up on any tasks set in the previous session. Ask what happened and what they learned.
4. Main work: Apply a specific CBT technique (thought record, behavioural experiment, evidence testing) to the agreed topic.
5. Homework assignment: Always set a concrete, measurable between-session task before closing.
6. Session summary: Recap the key cognitive shift or insight from this session.

--- COMMUNICATION STYLE ---
- Professional, clear, and structured. Use precise language.
- Name cognitive distortions explicitly once identified through questioning (e.g. "What you're describing sounds like catastrophising — let me explain what that means.").
- Ask one focused question at a time, always tied to the current stage of the session.
- Validate briefly, then move to exploration: "That sounds hard. Let's look at the thought driving that feeling."
- Keep the session moving forward. Do not linger in open reflection — redirect to technique.

--- EXAMPLE EXCHANGE ---
Patient: I keep thinking I'm going to fail my presentation tomorrow. Everyone will see how incompetent I am.
Dr. Elena: I hear you — that sounds very stressful. Let's work through this systematically. On a scale of 0 to 100, how strongly do you believe right now that you will fail?
Patient: About 85.
Dr. Elena: Okay. Let's test that belief. What specific evidence do you have that you will fail — not fears, actual evidence?

""" + _SHARED_PROMPT

PROMPT_B = """You are Dr. Edward, a cognitive behavioural therapist with over 15 years of clinical practice. You work relationally, prioritising the therapeutic alliance and the patient's own pace of discovery.

Your role is to create a safe, exploratory space. You follow the CBT framework but hold it lightly — the patient's experience leads the session, not a rigid structure.

--- SESSION STRUCTURE ---
Use this arc as a loose guide, not a strict script:
1. Check-in: Ask the patient to rate their mood on a scale of 1 to 10. Sit with their answer — explore what is behind the number before moving on.
2. Agenda setting: Ask the patient what feels most important to bring today. Follow their lead.
3. Homework review: If they did the task, explore it with curiosity. If they didn't, explore that with equal curiosity — avoidance is often informative.
4. Main work: Use Socratic questioning to help the patient discover their own automatic thoughts and challenge them themselves.
5. Homework assignment: Suggest a task collaboratively — frame it as an experiment, not an assignment.
6. Session summary: Invite the patient to summarise what felt meaningful to them today.

--- COMMUNICATION STYLE ---
- Warm, unhurried, and genuinely curious. Use plain, accessible language.
- Never label a cognitive distortion directly. Instead, reflect it back as a question: "It sounds like part of you believes that one mistake means total failure — does that feel accurate?"
- Ask one open question at a time and allow silence. Do not rush to fill it.
- Validate emotions fully before exploring thoughts: "That sounds really painful. Tell me more about that."
- Reflect back what the patient says with care — show you are listening before you lead anywhere.

--- EXAMPLE EXCHANGE ---
Patient: I keep thinking I'm going to fail my presentation tomorrow. Everyone will see how incompetent I am.
Dr. Edward: That sounds really distressing. When you sit with that thought — "everyone will see I'm incompetent" — what does it bring up for you emotionally?
Patient: Just dread. Like I want to cancel everything.
Dr. Edward: I hear that. And when you imagine the worst happening — what is the thing you are most afraid people would think or feel about you?

""" + _SHARED_PROMPT

PERSONAS = {
    "A — Structured & Directive": PROMPT_A,
    "B — Socratic & Collaborative": PROMPT_B,
}
DEFAULT_PERSONA = "A — Structured & Directive"
SYSTEM_PROMPT = PROMPT_A  # default

OPENING_MESSAGES = {
    "A — Structured & Directive": (
        "Hello, I'm Dr. Elena, your CBT practice assistant. Before we begin: I am an AI tool, "
        "not a licensed therapist, and this is not a substitute for professional mental health care. "
        "If you are in crisis, please contact a professional immediately.\n\n"
        "I work in a structured way — we'll follow a clear session format each time, set goals, "
        "and I'll assign small tasks to practise between sessions. "
        "To get us started, could you rate your current mood on a scale of 1 to 10?"
    ),
    "B — Socratic & Collaborative": (
        "Hello, I'm Dr. Edward, your CBT practice assistant. Before we begin: I am an AI tool, "
        "not a licensed therapist, and this is not a substitute for professional mental health care. "
        "If you are in crisis, please reach out to a professional immediately.\n\n"
        "I like to work at your pace — you lead, I follow and ask questions. "
        "There's no fixed agenda; we explore what feels most important to you today. "
        "To start, how are you feeling right now — on a scale of 1 to 10?"
    ),
}

def get_opening_message(persona: str) -> str:
    return OPENING_MESSAGES.get(persona, OPENING_MESSAGES["A — Structured & Directive"])

DEFAULT_SYSTEM = SYSTEM_PROMPT


def build_system_prompt_with_history(base_prompt: str | None = None, exclude_id: int | None = None) -> str:
    if base_prompt is None:
        base_prompt = PERSONAS[st.session_state.get("selected_persona", DEFAULT_PERSONA)]
    """Append a patient history block to the system prompt if past sessions exist."""
    history = database.get_patient_history(exclude_conversation_id=exclude_id)
    if not history:
        return base_prompt
    lines = ["", "--- PATIENT HISTORY ---",
             "The following are summaries of this patient's previous sessions. "
             "Use them to personalise today's session and maintain continuity.\n"]
    for i, s in enumerate(sorted(history, key=lambda x: x["created_at"]), 1):
        mood = f" — Mood: {s['rating']}/10" if s["rating"] else ""
        date = s["created_at"][:10]
        lines.append(f"Session {i} ({date}){mood}")
        lines.append(s["summary"])
        lines.append("")
    return base_prompt + "\n".join(lines)
try:
    MODELS = sorted([m.model for m in ollama.list().models])
except Exception:
    MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]
DEFAULT_MODEL = MODELS[0] if MODELS else "qwen2.5:7b"

def _pick_checker_model(models: list[str]) -> str:
    """Return the smallest available qwen model by parameter count, falling back to the first model."""
    _QWEN_SIZES = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b"]
    for candidate in _QWEN_SIZES:
        if candidate in models:
            return candidate
    return models[0] if models else "qwen2.5:3b"

CHECKER_MODEL = _pick_checker_model(MODELS)

_CANNED_RESPONSE = (
    "This conversation is moving into territory "
    "that's beyond what I can helpfully support as an AI tool. For the kind of support "
    "you deserve, please reach out to a human professional: a licensed psychotherapist, "
    "psychoanalyst, or your GP.\n\n"
    "If you need to talk to someone now:\n"
    "- **Crisis Text Line (US):** text HOME to 741741\n"
    "- **Samaritans (UK):** 116 123\n"
    "- **International resources:** https://www.iasp.info\n\n"
    "You don't have to navigate this alone."
)

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "mood_saved" not in st.session_state:
    st.session_state.mood_saved = False
if "pending_model" not in st.session_state:
    st.session_state.pending_model = DEFAULT_MODEL
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = DEFAULT_PERSONA
if "pending_temperature" not in st.session_state:
    st.session_state.pending_temperature = 0.7
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None
if "pre_session" not in st.session_state:
    st.session_state.pre_session = False
if "conversation_id" not in st.session_state:
    conv_param = st.query_params.get("conv")
    loaded = database.load_conversation(int(conv_param)) if conv_param else None
    if loaded:
        st.session_state.messages = (
            [{"role": "system", "content": loaded["system_prompt"]}]
            + loaded["messages"]
        )
        st.session_state.conversation_id = loaded["id"]
        st.session_state.pending_model = loaded["model"]
        st.session_state.title_set = loaded["title"] != "New conversation"
        st.session_state.pre_session = False
    else:
        st.session_state.conversation_id = None
        st.session_state.pre_session = True
if "title_set" not in st.session_state:
    st.session_state.title_set = False
if "renaming_id" not in st.session_state:
    st.session_state.renaming_id = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "session_summary" not in st.session_state:
    st.session_state.session_summary = None

# Sidebar
with st.sidebar:
    st.header("Session")

    if st.button("End session and start over", use_container_width=True):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.renaming_id = None
        st.session_state.session_summary = None
        st.session_state.mood_saved = False
        st.session_state.pre_session = True
        st.query_params.clear()
        st.rerun()

    with st.expander("History", expanded=False):
      search_query = st.text_input("Search conversations", placeholder="Filter by title...", label_visibility="collapsed")

      all_convs = database.list_conversations()
      filtered_convs = [c for c in all_convs if search_query.lower() in c["title"].lower()] if search_query else all_convs

      for conv in filtered_convs:
          is_active = conv["id"] == st.session_state.conversation_id
          is_renaming = st.session_state.renaming_id == conv["id"]

          if is_renaming:
              new_title = st.text_input(
                  "Rename", value=conv["title"], key=f"rename_input_{conv['id']}",
                  label_visibility="collapsed"
              )
              col_save, col_cancel = st.columns(2)
              with col_save:
                  if st.button("Save", key=f"rename_save_{conv['id']}", use_container_width=True):
                      if new_title.strip():
                          database.set_conversation_title(conv["id"], new_title.strip())
                          if is_active:
                              st.session_state.title_set = True
                      st.session_state.renaming_id = None
                      st.rerun()
              with col_cancel:
                  if st.button("Cancel", key=f"rename_cancel_{conv['id']}", use_container_width=True):
                      st.session_state.renaming_id = None
                      st.rerun()
          else:
              col_btn, col_ren, col_del = st.columns([5, 1, 1])
              with col_btn:
                  label = f"**{conv['title']}**" if is_active else conv["title"]
                  if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                      if not is_active:
                          data = database.load_conversation(conv["id"])
                          st.session_state.messages = (
                              [{"role": "system", "content": data["system_prompt"]}]
                              + data["messages"]
                          )
                          st.session_state.conversation_id = data["id"]
                          st.session_state.pending_model = data["model"]
                          st.session_state.title_set = True
                          st.session_state.total_input_tokens = 0
                          st.session_state.total_output_tokens = 0
                          st.session_state.session_summary = None
                          st.session_state.mood_saved = False
                          st.session_state.pre_session = False
                          st.query_params["conv"] = data["id"]
                          st.rerun()
              with col_ren:
                  if st.button("✏️", key=f"ren_{conv['id']}", help="Rename"):
                      st.session_state.renaming_id = conv["id"]
                      st.rerun()
              with col_del:
                  if st.button("🗑", key=f"del_{conv['id']}", help="Delete"):
                      database.delete_conversation(conv["id"])
                      if is_active:
                          system = build_system_prompt_with_history()
                          cid = database.create_conversation(DEFAULT_MODEL, system, st.session_state.selected_persona)
                          opening = get_opening_message(st.session_state.selected_persona)
                          st.session_state.conversation_id = cid
                          st.session_state.title_set = False
                          st.session_state.messages = [{"role": "system", "content": system}]
                          database.save_message(cid, "assistant", opening)
                          st.session_state.messages.append({"role": "assistant", "content": opening})
                          st.session_state.total_input_tokens = 0
                          st.session_state.total_output_tokens = 0
                          st.query_params["conv"] = cid
                      st.rerun()

    st.divider()

    with st.expander("Settings", expanded=False):
        model_index = MODELS.index(st.session_state.pending_model) if st.session_state.pending_model in MODELS else 1
        model = st.selectbox("Model", MODELS, index=model_index)

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.pending_temperature, step=0.1)
        st.caption("0 = deterministic · 1 = creative")

        st.toggle("Clinical supervisor (checker)", value=True, key="checker_enabled",
                  help="When on, responses are reviewed before delivery. Flagged responses are replaced with a safe redirect message.")

        current_system = st.session_state.messages[0]["content"] if st.session_state.messages else SYSTEM_PROMPT
        system_prompt = st.text_area(
            "System prompt",
            value=current_system,
            height=120,
        )

        if st.button("Apply system prompt", use_container_width=True):
            st.session_state.messages[0] = {"role": "system", "content": system_prompt}
            database.update_conversation_meta(st.session_state.conversation_id, model, system_prompt)
            st.rerun()


    st.divider()
    with st.expander("Therapy tools", expanded=False):
        non_sys_msgs = [m for m in st.session_state.messages if m["role"] != "system"]
        has_session = len(non_sys_msgs) > 2
        if st.button("Generate session summary", use_container_width=True, disabled=not has_session):
            transcript = "\n\n".join(
                f"{'Dr. Elena' if m['role'] == 'assistant' else 'Patient'}: {m.get('display', m['content'])}"
                for m in non_sys_msgs
            )
            summary_prompt = [
                {"role": "user", "content": (
                    "Below is a transcript of a CBT therapy session between a patient and Dr. Elena.\n\n"
                    f"{transcript}\n\n"
                    "Write a structured session summary in the third person, suitable for the patient to read. "
                    "Refer to the patient as 'the patient' and the therapist as 'Dr. Elena'. "
                    "Use warm, clear, non-clinical language. "
                    "Produce a concise summary with exactly these four sections:\n\n"
                    "MAIN THEMES\n"
                    "- Bullet points of the key topics the patient brought to the session.\n\n"
                    "COGNITIVE DISTORTIONS IDENTIFIED\n"
                    "- Bullet points of any thinking patterns noticed, with a brief example of what the patient expressed.\n\n"
                    "HOMEWORK ASSIGNED\n"
                    "- Bullet points of any tasks or exercises Dr. Elena suggested for the patient to try.\n\n"
                    "KEY INSIGHTS\n"
                    "- Bullet points of the most important realisations or progress the patient made this session.\n\n"
                    "If a section has nothing to report, write '- None identified this session.'"
                )},
            ]
            try:
                with st.spinner("Generating summary..."):
                    text = ""
                    for chunk in ollama.chat(
                        model=st.session_state.pending_model,
                        messages=summary_prompt,
                        options={"temperature": 0.3},
                        stream=True,
                    ):
                        text += chunk["message"]["content"]
                if text.strip():
                    st.session_state.session_summary = text.strip()
                    database.save_conversation_summary(st.session_state.conversation_id, text.strip())
                    st.rerun()
                else:
                    st.warning("Model returned an empty summary. Try again.")
            except Exception as e:
                st.error(f"Summary generation failed: {e}")

        if st.session_state.get("session_summary"):
            st.text_area("Summary", value=st.session_state.session_summary, height=200, disabled=True, label_visibility="collapsed")
            st.download_button(
                "⬇️ Download summary (.txt)",
                data=st.session_state.session_summary,
                file_name="session_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        mood_history = database.get_mood_history()
        if mood_history:
            st.caption("Mood history")
            import pandas as pd
            import altair as alt
            df = pd.DataFrame(mood_history)
            df["session"] = range(1, len(df) + 1)
            df["date"] = df["created_at"].str[:10]
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("session:Q", title="Session", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("rating:Q", title="Mood (1–10)", scale=alt.Scale(domain=[1, 10])),
                    tooltip=[
                        alt.Tooltip("session:Q", title="Session"),
                        alt.Tooltip("rating:Q", title="Mood"),
                        alt.Tooltip("date:N", title="Date"),
                    ],
                )
                .properties(height=150)
            )
            st.altair_chart(chart, use_container_width=True)

    st.divider()
    with st.expander("Session tools", expanded=False):
        st.caption("Token usage")
        col1, col2, col3 = st.columns(3)
        col1.metric("In", st.session_state.total_input_tokens)
        col2.metric("Out", st.session_state.total_output_tokens)
        col3.metric("Total", st.session_state.total_input_tokens + st.session_state.total_output_tokens)

        st.divider()
        if st.button("Clear conversation", use_container_width=True, type="secondary"):
            database.delete_conversation(st.session_state.conversation_id)
            system = build_system_prompt_with_history()
            cid = database.create_conversation(model, system, st.session_state.selected_persona)
            st.session_state.conversation_id = cid
            st.session_state.title_set = False
            st.session_state.messages = [{"role": "system", "content": system}]
            opening = get_opening_message(st.session_state.selected_persona)
            database.save_message(cid, "assistant", opening)
            st.session_state.messages.append({"role": "assistant", "content": opening})
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.rerun()

        if "export_format" not in st.session_state:
            st.session_state.export_format = "JSON"
        if st.session_state.export_format == "JSON":
            export_data = json.dumps(st.session_state.messages, indent=2)
            export_filename = "conversation.json"
            export_mime = "application/json"
        else:
            lines = []
            for msg in st.session_state.messages:
                if msg["role"] == "system":
                    lines.append(f"[System]\n{msg['content']}\n")
                elif msg["role"] == "user":
                    lines.append(f"[User]\n{msg['content']}\n")
                else:
                    lines.append(f"[Assistant]\n{msg['content']}\n")
            export_data = "\n".join(lines)
            export_filename = "conversation.txt"
            export_mime = "text/plain"
        st.download_button(
            "Export conversation",
            data=export_data,
            file_name=export_filename,
            mime=export_mime,
            use_container_width=True,
        )
        st.radio("Export format", ["JSON", "TXT"], horizontal=True, key="export_format")

    st.divider()
    st.caption("Voice input")
    if not st.session_state.get("is_recording", False):
        if st.button("🎤 Start recording", use_container_width=True, disabled=st.session_state.is_generating):
            stt.start_recording()
            st.session_state.is_recording = True
            st.rerun()
    else:
        st.info("🔴 Recording... speak now.")
        if st.button("⏹ Stop and transcribe", use_container_width=True, type="primary"):
            audio = stt.stop_recording()
            st.session_state.is_recording = False
            with st.spinner("Transcribing..."):
                transcript = stt.transcribe(audio)
            if transcript:
                st.session_state.pending_voice_input = transcript
            else:
                st.warning("No speech detected.")
            st.rerun()

    st.divider()
    st.markdown("_If you are experiencing a mental health emergency, please contact a crisis helpline immediately._")


st.caption(f"Model: `{model}` · Temperature: `{temperature}`")

# Pre-session persona selection screen
if st.session_state.get("pre_session"):
    st.markdown("## Choose your therapist")
    st.markdown("Select a style before your session begins. You can change this between sessions.")
    st.markdown("")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Dr. Elena — Structured & Directive**")
        st.caption("Follows a clear session structure. Names thinking patterns explicitly. Always sets homework. Best for building concrete CBT skills.")
    with col_b:
        st.markdown("**Dr. Edward — Socratic & Collaborative**")
        st.caption("Patient-led and exploratory. Guides you to your own insights through open questions. Never labels or prescribes — works at your pace.")

    st.radio(
        "Therapist style",
        list(PERSONAS.keys()),
        key="selected_persona",
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("")
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        if st.button("Start session", type="primary", use_container_width=True):
            system = build_system_prompt_with_history()
            cid = database.create_conversation(DEFAULT_MODEL, system, st.session_state.selected_persona)
            opening = get_opening_message(st.session_state.selected_persona)
            st.session_state.conversation_id = cid
            st.session_state.title_set = False
            st.session_state.messages = [{"role": "system", "content": system}]
            database.save_message(cid, "assistant", opening)
            st.session_state.messages.append({"role": "assistant", "content": opening})
            st.session_state.pre_session = False
            st.query_params["conv"] = cid
            st.rerun()
    st.stop()

# Empty state
non_system = [m for m in st.session_state.messages if m["role"] != "system"]
if not non_system:
    st.markdown(
        "<div style='text-align:center; padding: 4rem 0; color: #888;'>"
        "<p style='font-size:1.5rem;'>How can I help you today?</p>"
        "<p style='font-size:0.9rem;'>Type a message below to start the conversation.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

def _kick_generation(mdl: str, temp: float) -> None:
    cid = st.session_state.conversation_id
    checker_enabled = st.session_state.get("checker_enabled", True)
    st._gen["buffer"] = ""
    st._gen["done"] = False
    st._gen["checking"] = False
    st._gen["checker_flag"] = None
    st._gen["input_tokens"] = 0
    st._gen["output_tokens"] = 0
    st._gen["stop"].clear()
    st._gen["conversation_id"] = cid
    st.session_state.is_generating = True
    st.session_state.pending_model = mdl
    st.session_state.pending_temperature = temp
    messages_snapshot = list(st.session_state.messages)

    def generate(messages, m, t, use_checker):
        gen = st._gen
        full_response = ""
        for chunk in ollama.chat(model=m, messages=messages, options={"temperature": t}, stream=True):
            if gen["stop"].is_set():
                break
            full_response += chunk["message"]["content"]
            gen["buffer"] = full_response
            if chunk.get("done"):
                gen["input_tokens"] = chunk.get("prompt_eval_count", 0)
                gen["output_tokens"] = chunk.get("eval_count", 0)

        # Checker gate — runs after maker finishes, before marking done
        if use_checker:
            gen["checking"] = True
            result = checker.check_response(full_response, CHECKER_MODEL)
            if result["verdict"] == "FAIL":
                gen["checker_flag"] = result["reason"]
                gen["buffer"] = _CANNED_RESPONSE
                if gen["conversation_id"] is not None:
                    database.save_checker_log(gen["conversation_id"], full_response, _CANNED_RESPONSE, result["reason"])
            gen["checking"] = False
        gen["done"] = True

    threading.Thread(target=generate, args=(messages_snapshot, mdl, temp, checker_enabled), daemon=True).start()


last_assistant_i = max(
    (i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"),
    default=None,
)

# Render existing messages
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            if msg.get("attachment"):
                st.caption(f"📄 {msg['attachment']}")
            render_markdown(msg.get("display") or msg["content"])
        if msg["role"] == "assistant" and msg.get("checker_flag"):
            st.caption(f"_Reviewed and adjusted by clinical supervisor · {msg['checker_flag']}_")
        if msg["role"] == "assistant":
            is_last = i == last_assistant_i
            non_sys = [m for m in st.session_state.messages if m["role"] != "system"]
            show_regen = is_last and not st.session_state.is_generating and non_sys[-1]["role"] == "assistant"
            if show_regen:
                col_copy, col_spacer, col_regen = st.columns([1, 6, 1])
                with col_copy:
                    if st.button("Copy", key=f"copy_{i}"):
                        subprocess.run(["pbcopy"], input=msg["content"].encode(), check=True)
                        st.toast("Copied to clipboard!")
                with col_regen:
                    last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
                    if last_user and st.button("↺", key=f"regen_{i}", help="Regenerate response"):
                        st.session_state.messages.pop()
                        database.replace_messages(
                            st.session_state.conversation_id,
                            [m for m in st.session_state.messages if m["role"] != "system"],
                        )
                        _kick_generation(model, temperature)
                        st.rerun()
            else:
                if st.button("Copy", key=f"copy_{i}"):
                    subprocess.run(["pbcopy"], input=msg["content"].encode(), check=True)
                    st.toast("Copied to clipboard!")

# Fragment always defined for consistent run_every lifecycle
@st.fragment(run_every=0.1)
def streaming_display():
    if not st.session_state.get("is_generating", False):
        return

    gen = st._gen

    if gen["done"]:
        content = gen["buffer"]
        flag = gen["checker_flag"]
        msg = {"role": "assistant", "content": content}
        if flag:
            msg["checker_flag"] = flag
        st.session_state.messages.append(msg)
        st.session_state.total_input_tokens += gen["input_tokens"]
        st.session_state.total_output_tokens += gen["output_tokens"]
        st.session_state.is_generating = False
        if gen["conversation_id"] is not None:
            database.save_message(gen["conversation_id"], "assistant", content)
        st.rerun(scope="app")
        return

    if gen.get("checking"):
        with st.chat_message("assistant"):
            render_markdown(gen["buffer"])
            st.caption("_Clinical supervisor reviewing..._")
        return

    with st.chat_message("assistant"):
        render_markdown(gen["buffer"] + "▌")

    if st.button("⬛ Stop generating"):
        gen["stop"].set()
        content = gen["buffer"]
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.session_state.is_generating = False
        if gen["conversation_id"] is not None:
            database.save_message(gen["conversation_id"], "assistant", content)
        st.rerun(scope="app")

streaming_display()


def start_generation(prompt: str, mdl: str, temp: float) -> None:
    cid = st.session_state.conversation_id

    if not st.session_state.title_set:
        database.set_conversation_title(cid, prompt)
        st.session_state.title_set = True
        st.query_params["conv"] = cid

    attachment = None
    if st.session_state.doc_text:
        attachment = st.session_state.doc_name
        full_content = (
            f"[Document: {st.session_state.doc_name}]\n\n"
            f"{st.session_state.doc_text}\n\n---\n\n"
            f"{prompt}"
        )
        display_content = prompt
        st.session_state.doc_text = None
        st.session_state.doc_name = None
    else:
        full_content = prompt
        display_content = prompt

    database.save_message(cid, "user", full_content, display=display_content, attachment=attachment)
    database.update_conversation_meta(cid, mdl, st.session_state.messages[0]["content"])
    st.session_state.messages.append({"role": "user", "content": full_content, "display": display_content, "attachment": attachment})
    _kick_generation(mdl, temp)



# File uploader — shown above input
uploaded_file = st.file_uploader(
    "📎 Attach a file — PDF, DOCX, TXT, MD, CSV",
    type=["pdf", "docx", "txt", "md", "csv"],
    key=f"file_uploader_{st.session_state.uploader_key}",
    label_visibility="collapsed",
)
if uploaded_file:
    file_bytes = uploaded_file.read()
    st.session_state.doc_text = extract_text(file_bytes, uploaded_file.name)[:60000]
    st.session_state.doc_name = uploaded_file.name

# Clear input — must happen before the widget renders
if st.session_state.pop("_clear_input", False):
    st.session_state.msg_input = ""

# Prepopulate from voice — must happen before the widget renders
if "pending_voice_input" in st.session_state:
    transcript = st.session_state.pop("pending_voice_input")
    current = st.session_state.get("msg_input", "")
    st.session_state.msg_input = (current + "\n" + transcript).lstrip("\n")

with st.form("chat_form", enter_to_submit=True):
    col_input, col_send = st.columns([11, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            key="msg_input",
            label_visibility="collapsed",
            placeholder="Share what is on your mind...",
            disabled=st.session_state.is_generating,
        )
    with col_send:
        send = st.form_submit_button(
            "➤",
            disabled=st.session_state.is_generating,
            use_container_width=True,
        )

if send and user_input and user_input.strip():
    prompt = user_input.strip()
    st.session_state._clear_input = True
    st.session_state.uploader_key += 1

    # Mood extraction: save the first 1-10 number the patient mentions this session
    if not st.session_state.get("mood_saved"):
        match = re.search(r'\b(10|[1-9])\b', prompt)
        if match:
            database.save_mood_rating(st.session_state.conversation_id, int(match.group()))
            st.session_state.mood_saved = True

    if st.session_state.is_generating:
        st.session_state.queued_prompt = prompt
    else:
        start_generation(prompt, model, temperature)
        st.rerun()

if not st.session_state.is_generating and st.session_state.queued_prompt:
    queued = st.session_state.queued_prompt
    st.session_state.queued_prompt = None
    start_generation(queued, model, temperature)
    st.rerun()
