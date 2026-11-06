"""
Abdur AI Assistant - an AI voice assistant built with Streamlit + Google Gemini.

Flow (voice): microphone -> Gemini understands the speech and answers ->
Gemini TTS generates a real WAV voice note -> auto-played + written transcript.

Secrets: set GEMINI_API_KEY in Streamlit Secrets or as an environment variable.
"""

import base64
import hashlib
import html
import io
import json
import os
import re
import time
import uuid
import wave
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Abdur AI Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_NAME = "Abdur AI Assistant"

# Text / audio-understanding models (tried in order).
TEXT_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
# Text-to-speech models (tried in order).
TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-pro-preview-tts",
]

VOICES = [
    "Kore", "Puck", "Zephyr", "Aoede", "Charon", "Fenrir",
    "Leda", "Orus", "Callirhoe", "Autonoe", "Algieba", "Sulafat",
]
LANGUAGES = ["Auto-detect", "English", "Urdu"]
STYLES = {
    "Friendly": "warm, friendly and conversational",
    "Professional": "polished, clear and professional",
    "Concise": "extremely concise and direct",
}

MAX_TTS_CHARS = 2500   # characters spoken per answer (full text is always shown)
TTS_CHUNK_CHARS = 800  # each TTS request handles at most this many characters
MAX_HISTORY_MESSAGES = 24

URDU_RE = re.compile(r"[\u0600-\u06FF]")

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "transcript": {"type": "STRING"},
        "reply": {"type": "STRING"},
        "language": {"type": "STRING"},
    },
    "required": ["transcript", "reply", "language"],
}

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');

:root{
  --bg:#080B18; --bg2:#11152B; --purple:#9333EA; --violet:#A855F7;
  --cyan:#22D3EE; --white:#F8FAFC; --muted:#94A3B8;
}
html, body, .stApp, [class*="css"]{ font-family:'Inter',sans-serif; }
.stApp{
  background:
    radial-gradient(900px 500px at 70% -10%, rgba(147,51,234,.25), transparent 60%),
    radial-gradient(700px 400px at 0% 100%, rgba(34,211,238,.10), transparent 60%),
    var(--bg);
  color:var(--white);
}
header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:1.6rem; max-width:980px; }

section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0d1124 0%, #0a0e1f 100%);
  border-right:1px solid rgba(168,85,247,.18);
}
section[data-testid="stSidebar"] *{ color:var(--white); }

.brand{ display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1rem 0; }
.brand .dot{ width:38px; height:38px; border-radius:50%;
  background:radial-gradient(circle at 30% 30%, #fff, var(--cyan) 30%, var(--purple) 75%);
  box-shadow:0 0 22px rgba(168,85,247,.8); }
.brand .name{ font-weight:700; font-size:1.08rem; line-height:1.1; }
.brand .sub{ color:var(--muted); font-size:.74rem; }

.chip{ display:inline-block; padding:.22rem .7rem; border-radius:999px; font-size:.76rem;
  border:1px solid rgba(168,85,247,.35); background:rgba(147,51,234,.12); }
.chip.ok{ border-color:rgba(34,211,238,.5); background:rgba(34,211,238,.10); color:#a5f3fc; }
.chip.bad{ border-color:rgba(248,113,113,.5); background:rgba(248,113,113,.10); color:#fecaca; }

h1,h2,h3{ color:var(--white) !important; letter-spacing:-.01em; }
.page-title{ font-size:1.7rem; font-weight:700; margin:0 0 .2rem 0;
  background:linear-gradient(90deg,var(--white),var(--violet) 60%,var(--cyan));
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.page-sub{ color:var(--muted); margin-bottom:1rem; }

/* Orb */
.orb-wrap{ display:flex; flex-direction:column; align-items:center; margin:.4rem 0 .6rem 0; }
.orb{ position:relative; width:170px; height:170px; border-radius:50%;
  background:radial-gradient(circle at 32% 28%, #ffffff 0%, var(--cyan) 18%, var(--violet) 52%, #3b0a77 100%);
  box-shadow:0 0 40px rgba(168,85,247,.65), 0 0 110px rgba(147,51,234,.4), inset 0 0 40px rgba(255,255,255,.25);
  animation:breathe 4.5s ease-in-out infinite; }
.orb .ring{ position:absolute; inset:-14px; border-radius:50%;
  border:1.5px solid rgba(168,85,247,.45); animation:ripple 3.2s ease-out infinite; }
.orb .ring.r2{ animation-delay:1.6s; }
.orb.thinking{ animation:breathe 1.3s ease-in-out infinite;
  filter:hue-rotate(-25deg) saturate(1.2); }
.orb.thinking::after{ content:""; position:absolute; inset:-10px; border-radius:50%;
  border:3px solid transparent; border-top-color:var(--cyan); border-right-color:var(--violet);
  animation:spin 1s linear infinite; }
.orb.speaking{ animation:speak .55s ease-in-out infinite; }
.orb.speaking .ring{ animation-duration:1.2s; border-color:rgba(34,211,238,.6); }
.orb-label{ margin-top:1.1rem; color:var(--muted); font-size:.92rem; letter-spacing:.04em; }
.orb-label.thinking{ color:var(--cyan); } .orb-label.speaking{ color:var(--violet); }
@keyframes breathe{ 0%,100%{ transform:scale(1);} 50%{ transform:scale(1.045);} }
@keyframes speak{ 0%,100%{ transform:scale(1);} 50%{ transform:scale(1.09);} }
@keyframes ripple{ 0%{ transform:scale(.95); opacity:.8;} 100%{ transform:scale(1.5); opacity:0;} }
@keyframes spin{ to{ transform:rotate(360deg);} }

/* Native audio recorder styled like a mic control */
div[data-testid="stAudioInput"]{ max-width:420px; margin:0 auto; }
div[data-testid="stAudioInput"] > div{ border-radius:999px;
  background:rgba(147,51,234,.14); border:1px solid rgba(168,85,247,.5);
  box-shadow:0 0 26px rgba(147,51,234,.35); }
div[data-testid="stAudioInput"] label{ display:none; }
.mic-hint{ text-align:center; color:var(--muted); font-size:.85rem; margin:.4rem 0 1.2rem 0; }

/* Messages */
.msg{ border-radius:16px; padding:.85rem 1.05rem; margin:.65rem 0 .25rem 0;
  border:1px solid rgba(168,85,247,.2); background:var(--bg2); line-height:1.55; }
.msg.user{ background:linear-gradient(135deg, rgba(147,51,234,.28), rgba(168,85,247,.12));
  border-color:rgba(168,85,247,.45); margin-left:12%; }
.msg.ai{ margin-right:12%; border-left:3px solid var(--cyan); }
.msg .who{ font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.25rem; }
.msg.rtl{ direction:rtl; text-align:right; font-family:'Noto Nastaliq Urdu','Inter',serif;
  line-height:2.25; font-size:1.08rem; }
.msg.rtl .who{ direction:ltr; text-align:left; font-family:'Inter',sans-serif; }

.card{ background:var(--bg2); border:1px solid rgba(168,85,247,.22); border-radius:16px;
  padding:1rem 1.2rem; margin:.5rem 0; }
.card h4{ margin:.1rem 0 .4rem 0; }
.card ol, .card ul{ margin:.2rem 0 .2rem 1.1rem; color:#cbd5e1; }

/* Buttons */
.stButton > button, .stDownloadButton > button{
  border-radius:12px; border:1px solid rgba(168,85,247,.45);
  background:linear-gradient(135deg, rgba(147,51,234,.55), rgba(168,85,247,.25));
  color:var(--white); transition:all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover{
  border-color:var(--cyan); box-shadow:0 0 18px rgba(34,211,238,.35); transform:translateY(-1px); color:#fff; }

div[data-testid="stChatInput"]{ border-radius:16px; }
[data-testid="stMetric"]{ background:var(--bg2); border:1px solid rgba(168,85,247,.2);
  border-radius:14px; padding:.7rem .9rem; }

@media (max-width:700px){
  .block-container{ padding:1rem .8rem; }
  .msg.user{ margin-left:3%; } .msg.ai{ margin-right:3%; }
  .orb{ width:130px; height:130px; }
  .page-title{ font-size:1.35rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Audio player component (real playback, replay, stop, download)
# --------------------------------------------------------------------------- #
PLAYER_TPL = """
<style>
 body{margin:0;font-family:Inter,Arial,sans-serif;background:transparent;}
 .p{display:flex;align-items:center;gap:8px;padding:8px 12px;border-radius:14px;
    background:#11152B;border:1px solid rgba(168,85,247,.35);color:#F8FAFC;font-size:13px;flex-wrap:wrap;}
 button{cursor:pointer;border:1px solid rgba(168,85,247,.6);background:linear-gradient(135deg,#9333EA,#A855F7);
    color:#fff;border-radius:50%;width:34px;height:34px;font-size:14px;}
 button.s{background:#1b2040;}
 input[type=range]{flex:1;min-width:90px;accent-color:#22D3EE;}
 a{color:#22D3EE;text-decoration:none;font-size:12px;border:1px solid rgba(34,211,238,.5);
    padding:5px 10px;border-radius:999px;}
 #t{min-width:78px;color:#94A3B8;font-size:12px;}
</style>
<div class="p">
  <button id="pp" title="Play / pause">&#9654;</button>
  <input id="seek" type="range" min="0" max="100" value="0" step="0.1">
  <span id="t">0:00</span>
  <button id="rp" class="s" title="Replay">&#10227;</button>
  <button id="sb" class="s" title="Stop">&#9632;</button>
  <a id="dl" download="abdur-voice-note-__UID__.wav" href="data:audio/wav;base64,__B64__">&#11015; WAV</a>
</div>
<audio id="a" preload="auto" src="data:audio/wav;base64,__B64__"></audio>
<script>
const a=document.getElementById('a'),pp=document.getElementById('pp'),seek=document.getElementById('seek'),
      t=document.getElementById('t'),rp=document.getElementById('rp'),sb=document.getElementById('sb');
const SPEED=__SPEED__;
function fmt(s){s=Math.floor(s||0);return Math.floor(s/60)+':'+String(s%60).padStart(2,'0');}
a.playbackRate=SPEED;
pp.onclick=()=>{a.playbackRate=SPEED;a.paused?a.play():a.pause();};
a.onplay=()=>{pp.innerHTML='&#10074;&#10074;';};
a.onpause=()=>{pp.innerHTML='&#9654;';};
a.onended=()=>{pp.innerHTML='&#9654;';};
a.onloadedmetadata=()=>{t.textContent='0:00 / '+fmt(a.duration);};
a.ontimeupdate=()=>{seek.value=a.duration?(a.currentTime/a.duration*100):0;
  t.textContent=fmt(a.currentTime)+' / '+fmt(a.duration);};
seek.oninput=()=>{if(a.duration){a.currentTime=seek.value/100*a.duration;}};
rp.onclick=()=>{a.currentTime=0;a.playbackRate=SPEED;a.play();};
sb.onclick=()=>{a.pause();a.currentTime=0;};
a.onerror=()=>{t.textContent='Playback failed';};
if(__AUTO__){a.playbackRate=SPEED;a.play().catch(()=>{t.textContent='Tap \\u25B6 to play';});}
</script>
"""


def render_player(wav: bytes, uid: str, autoplay: bool) -> None:
    speed = float(st.session_state.get("voice_speed", 1.0))
    page = (
        PLAYER_TPL.replace("__B64__", base64.b64encode(wav).decode())
        .replace("__SPEED__", str(speed))
        .replace("__AUTO__", "true" if autoplay else "false")
        .replace("__UID__", uid[:8])
    )
    components.html(page, height=64)


def stop_all_audio() -> None:
    """Pause every audio player on the page (best effort)."""
    components.html(
        "<script>try{window.parent.document.querySelectorAll('iframe').forEach(function(f){"
        "try{f.contentDocument.querySelectorAll('audio').forEach(function(a){a.pause();a.currentTime=0;});}"
        "catch(e){}});}catch(e){}</script>",
        height=0,
    )


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def new_conversation() -> str:
    cid = uuid.uuid4().hex
    st.session_state.conversations[cid] = {
        "title": "New conversation",
        "created": datetime.now().strftime("%d %b %Y, %H:%M"),
        "messages": [],
    }
    st.session_state.current = cid
    return cid


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("conversations", {})
    ss.setdefault("page", "🎙️ Voice Assistant")
    ss.setdefault("mic_n", 0)
    ss.setdefault("autoplay_id", None)
    ss.setdefault("flash", None)
    ss.setdefault("api_status", "unknown")
    ss.setdefault("user_key", "")
    ss.setdefault("stats", {"requests": 0, "voices": 0, "audio_sec": 0.0, "errors": 0})
    if "current" not in ss or ss.current not in ss.conversations:
        if ss.conversations:
            ss.current = list(ss.conversations)[-1]
        else:
            new_conversation()


def current_conv() -> dict:
    return st.session_state.conversations[st.session_state.current]


# --------------------------------------------------------------------------- #
# Gemini helpers
# --------------------------------------------------------------------------- #
def get_api_key() -> str:
    if st.session_state.get("user_key"):
        return st.session_state.user_key.strip()
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


@st.cache_resource(show_spinner=False)
def make_client(key: str):
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=90_000))


class AppError(Exception):
    """Error with a message that is safe to show to the user."""


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if isinstance(e, AppError):
        return msg
    if "api_key_invalid" in low or "api key not valid" in low or "invalid api key" in low:
        return "Your Gemini API key is invalid. Check GEMINI_API_KEY and try again."
    if "permission_denied" in low or "403" in low:
        return "Gemini refused the request (403). The key may lack access to this model or region."
    if "429" in low or "resource_exhausted" in low or "quota" in low:
        return "Gemini rate limit or quota reached. Wait a minute and try again."
    if any(x in low for x in ("timed out", "timeout", "connection", "getaddrinfo", "network", "unreachable")):
        return "Network problem while contacting Gemini. Check your connection and try again."
    if "503" in low or "unavailable" in low or "overloaded" in low or "500" in low:
        return "Gemini is busy right now. Please try again in a moment."
    if "safety" in low or "blocked" in low:
        return "Gemini could not answer that request because of its safety filters."
    return f"Gemini error: {msg[:300]}"


def is_transient(e: Exception) -> bool:
    low = str(e).lower()
    return any(x in low for x in ("503", "500", "unavailable", "overloaded", "timed out", "timeout",
                                  "connection", "reset", "network"))


def is_model_missing(e: Exception) -> bool:
    low = str(e).lower()
    return "404" in low or "not found" in low or "not supported" in low or "is not found" in low


def call_models(models, fn, tries: int = 3):
    """Run fn(model) over fallback models, retrying transient failures."""
    last = None
    for model in models:
        for attempt in range(tries):
            try:
                return fn(model)
            except Exception as e:  # noqa: BLE001
                last = e
                if is_model_missing(e):
                    break  # try next model
                if is_transient(e) and attempt < tries - 1:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                if is_transient(e):
                    break  # next model
                raise
    raise last if last else AppError("No model available.")


def build_system(voice: bool) -> str:
    lang = st.session_state.get("lang", "Auto-detect")
    style = STYLES.get(st.session_state.get("style", "Friendly"), STYLES["Friendly"])
    if lang == "Auto-detect":
        lang_rule = ("Reply in the same language the user speaks or writes (English or Urdu). "
                     "If the user speaks Roman Urdu, reply in Urdu script.")
    else:
        lang_rule = f"Always reply in {lang}, whatever language the user uses."
    rules = [
        f"You are {APP_NAME}, a {style} AI assistant that talks with people like a real person.",
        lang_rule,
        "When replying in Urdu, use proper Urdu script (Arabic-based, اردو) ONLY. "
        "Never use Devanagari/Hindi script and never use Roman Urdu in your reply.",
        "Write plain text only: no markdown, no asterisks, no bullet symbols, no emojis, "
        "because your reply will be read aloud.",
        "Use the earlier conversation for context.",
    ]
    if voice:
        rules.append("Keep replies natural and spoken-style, usually 1 to 4 sentences unless the user asks for detail.")
        rules.append(
            "The user's latest message is an audio recording. Transcribe exactly what they said in the "
            "language they spoke (Urdu in Urdu script), then answer. If the audio is silent or unintelligible, "
            "return an empty transcript and a short reply asking them to try again."
        )
        rules.append('Return JSON: {"transcript": ..., "reply": ..., "language": "English" or "Urdu"}.')
    else:
        rules.append("Give helpful, accurate answers. Be thorough when needed but avoid padding.")
    return "\n".join(rules)


def history_contents(conv: dict) -> list:
    out = []
    for m in conv["messages"][-MAX_HISTORY_MESSAGES:]:
        if not m["text"].strip():
            continue
        role = "user" if m["role"] == "user" else "model"
        out.append(types.Content(role=role, parts=[types.Part.from_text(text=m["text"])]))
    return out


def gen_config(system: str, json_out: bool = False) -> types.GenerateContentConfig:
    kwargs = dict(system_instruction=system, temperature=0.7,
                  thinking_config=types.ThinkingConfig(thinking_budget=0))
    if json_out:
        kwargs.update(response_mime_type="application/json", response_schema=RESPONSE_SCHEMA)
    return types.GenerateContentConfig(**kwargs)


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise AppError("Gemini returned an unreadable answer. Please try again.")


def voice_turn(client, audio: bytes, mime: str, conv: dict) -> dict:
    contents = history_contents(conv)
    contents.append(types.Content(role="user", parts=[
        types.Part.from_bytes(data=audio, mime_type=mime),
        types.Part.from_text(text="Transcribe my voice message and reply."),
    ]))
    cfg = gen_config(build_system(voice=True), json_out=True)
    resp = call_models(TEXT_MODELS, lambda m: client.models.generate_content(model=m, contents=contents, config=cfg))
    data = parse_json(resp.text)
    return {
        "transcript": str(data.get("transcript", "")).strip(),
        "reply": str(data.get("reply", "")).strip(),
    }


def text_turn(client, conv: dict, user_text: str) -> str:
    contents = history_contents(conv)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
    cfg = gen_config(build_system(voice=False))
    resp = call_models(TEXT_MODELS, lambda m: client.models.generate_content(model=m, contents=contents, config=cfg))
    out = (resp.text or "").strip()
    if not out:
        raise AppError("Gemini returned an empty answer. Please rephrase and try again.")
    return out


# ---- Speech generation ----------------------------------------------------- #
def clean_for_speech(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>~]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_TTS_CHARS:
        cut = text[:MAX_TTS_CHARS]
        end = max(cut.rfind(". "), cut.rfind("۔"), cut.rfind("؟"), cut.rfind("? "), cut.rfind("! "))
        text = cut[: end + 1] if end > 200 else cut
    return text


def chunk_text(text: str) -> list:
    parts = re.split(r"(?<=[.!?۔؟])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > TTS_CHUNK_CHARS and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur.strip())
    return chunks or [text]


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def synth_chunk(client, model: str, text: str, voice: str):
    prompt = f"Say in a warm, natural, conversational tone:\n{text}"
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    try:
        part = resp.candidates[0].content.parts[0]
        data = part.inline_data.data
        mime = part.inline_data.mime_type or ""
    except Exception:
        raise AppError("The speech model returned no audio.")
    if not data:
        raise AppError("The speech model returned empty audio.")
    m = re.search(r"rate=(\d+)", mime)
    return data, int(m.group(1)) if m else 24000


def generate_speech(client, text: str):
    """Return (wav_bytes, duration_seconds). Raises on failure."""
    speech = clean_for_speech(text)
    if not speech:
        raise AppError("There is no text to speak.")
    voice = st.session_state.get("voice", "Kore")
    pcm_all, rate = b"", 24000
    for chunk in chunk_text(speech):
        data, rate = call_models(TTS_MODELS, lambda m: synth_chunk(client, m, chunk, voice))
        pcm_all += data
    wav = pcm_to_wav(pcm_all, rate)
    return wav, len(pcm_all) / (2 * rate)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def orb(state: str = "idle", label: str = "Ready when you are") -> str:
    return (f'<div class="orb-wrap"><div class="orb {state}"><div class="ring"></div>'
            f'<div class="ring r2"></div></div><div class="orb-label {state}">{html.escape(label)}</div></div>')


def msg_html(role: str, text: str) -> str:
    rtl = " rtl" if URDU_RE.search(text) else ""
    cls = "user" if role == "user" else "ai"
    who = "You" if role == "user" else APP_NAME
    body = html.escape(text).replace("\n", "<br>")
    return f'<div class="msg {cls}{rtl}"><div class="who">{who}</div>{body}</div>'


def add_message(conv: dict, role: str, text: str, audio=None, duration: float = 0.0) -> dict:
    m = {"id": uuid.uuid4().hex, "role": role, "text": text, "audio": audio, "duration": duration}
    conv["messages"].append(m)
    if role == "user" and conv["title"] == "New conversation" and text.strip():
        t = text.strip().replace("\n", " ")
        conv["title"] = (t[:38] + "…") if len(t) > 38 else t
    return m


def attach_speech(client, conv: dict, msg: dict) -> str:
    """Generate speech for a message. Returns '' or an error string."""
    try:
        wav, dur = generate_speech(client, msg["text"])
        msg["audio"], msg["duration"] = wav, dur
        st.session_state.stats["voices"] += 1
        st.session_state.stats["audio_sec"] += dur
        st.session_state.autoplay_id = msg["id"]
        return ""
    except Exception as e:  # noqa: BLE001
        st.session_state.stats["errors"] += 1
        return friendly_error(e)


def render_messages(conv: dict, client) -> None:
    autoplay_id = st.session_state.autoplay_id
    for m in conv["messages"]:
        st.markdown(msg_html(m["role"], m["text"]), unsafe_allow_html=True)
        if m["role"] != "ai" and m["role"] != "assistant":
            continue
        if m.get("audio"):
            render_player(m["audio"], m["id"], autoplay=(m["id"] == autoplay_id))
        elif client is not None:
            if st.button("🔊 Generate voice", key=f"gen_{m['id']}"):
                with st.spinner("Generating voice…"):
                    err = attach_speech(client, conv, m)
                if err:
                    st.session_state.flash = ("error", f"Audio generation failed: {err}")
                st.rerun()
    st.session_state.autoplay_id = None  # play once only


def show_flash() -> None:
    f = st.session_state.flash
    if f:
        kind, text = f
        {"error": st.error, "warning": st.warning, "success": st.success}.get(kind, st.info)(text)
        st.session_state.flash = None


def api_card() -> None:
    st.markdown(
        '<div class="card"><h4>🔑 Gemini API key needed</h4>'
        "Add <b>GEMINI_API_KEY</b> in Streamlit Secrets (or as an environment variable) and reload. "
        "You can also paste a key below for this browser session only - it is never saved.</div>",
        unsafe_allow_html=True,
    )
    k = st.text_input("Gemini API key", type="password", key="key_input", placeholder="AIza…")
    if st.button("Use this key", key="use_key") and k.strip():
        st.session_state.user_key = k.strip()
        st.session_state.api_status = "unknown"
        st.rerun()


def welcome_card() -> None:
    st.markdown(
        '<div class="card"><h4>👋 Welcome - how to talk to me</h4><ol>'
        "<li>Tap the microphone and allow microphone access when your browser asks.</li>"
        "<li>Speak naturally in English or Urdu, then tap the button again to stop.</li>"
        "<li>I will understand you, answer, and speak the answer back automatically.</li>"
        "<li>Use the player to replay, stop, or download the voice note.</li></ol></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_voice(client) -> None:
    st.markdown('<div class="page-title">Voice Assistant</div>'
                '<div class="page-sub">Speak naturally - I will answer out loud.</div>', unsafe_allow_html=True)
    show_flash()
    conv = current_conv()

    if client is None:
        st.markdown(orb("idle", "Waiting for API key"), unsafe_allow_html=True)
        api_card()
        return

    speaking = st.session_state.autoplay_id is not None
    orb_ph = st.empty()
    orb_ph.markdown(orb("speaking", "Speaking…") if speaking else orb("idle", "Tap the mic and start talking"),
                    unsafe_allow_html=True)

    audio = st.audio_input("Record your question", key=f"mic_{st.session_state.mic_n}")
    st.markdown('<div class="mic-hint">Tap to record · tap again to stop · I answer automatically. '
                "If nothing records, allow microphone access in your browser's address bar.</div>",
                unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("⏹ Stop speaking", key="stop_speaking"):
        stop_all_audio()
    if c2.button("➕ New chat", key="new_chat_voice"):
        new_conversation()
        st.rerun()

    if audio is not None:
        data = audio.getvalue()
        st.session_state.mic_n += 1  # resets the recorder after this run
        if len(data) < 3000:
            st.session_state.flash = ("warning", "I did not hear anything. Please record again and speak for a moment.")
            st.rerun()
        orb_ph.markdown(orb("thinking", "Listening to your words…"), unsafe_allow_html=True)
        try:
            st.session_state.stats["requests"] += 1
            result = voice_turn(client, data, audio.type or "audio/wav", conv)
            if not result["transcript"]:
                st.session_state.flash = ("warning", "I could not understand the recording. Please try again, a little closer to the microphone.")
                st.rerun()
            add_message(conv, "user", result["transcript"])
            ai_msg = add_message(conv, "ai", result["reply"] or "Sorry, I could not think of an answer.")
            orb_ph.markdown(orb("thinking", "Creating my voice…"), unsafe_allow_html=True)
            err = attach_speech(client, conv, ai_msg)
            if err:
                st.session_state.flash = ("error", f"I answered in text, but voice generation failed: {err}")
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in ("RerunException", "StopException"):
                raise
            st.session_state.stats["errors"] += 1
            st.session_state.api_status = "error"
            st.session_state.flash = ("error", friendly_error(e))
        st.rerun()

    st.markdown("### Conversation")
    if not conv["messages"]:
        welcome_card()
    else:
        render_messages(conv, client)


def page_text(client) -> None:
    st.markdown('<div class="page-title">Text Chat</div>'
                '<div class="page-sub">Type if you prefer - I can still speak the answers.</div>',
                unsafe_allow_html=True)
    show_flash()
    conv = current_conv()
    if client is None:
        api_card()
        return

    speak = st.toggle("🔊 Speak replies", value=True, key="speak_text")
    render_messages(conv, client)
    prompt = st.chat_input("Type your message (English or اردو)…")
    if prompt and prompt.strip():
        try:
            st.session_state.stats["requests"] += 1
            with st.spinner("Thinking…"):
                reply = text_turn(client, conv, prompt.strip())
            add_message(conv, "user", prompt.strip())
            ai_msg = add_message(conv, "ai", reply)
            if speak:
                with st.spinner("Creating voice…"):
                    err = attach_speech(client, conv, ai_msg)
                if err:
                    st.session_state.flash = ("error", f"Voice generation failed: {err}")
        except Exception as e:  # noqa: BLE001
            st.session_state.stats["errors"] += 1
            st.session_state.flash = ("error", friendly_error(e))
        st.rerun()


def page_notes(client) -> None:
    st.markdown('<div class="page-title">Voice Notes</div>'
                '<div class="page-sub">Every generated voice response, with transcript and download.</div>',
                unsafe_allow_html=True)
    notes = []
    for cid, c in st.session_state.conversations.items():
        for m in c["messages"]:
            if m.get("audio"):
                notes.append((c["title"], m))
    if not notes:
        st.info("No voice notes yet. Ask something in Voice Assistant or Text Chat and the audio will appear here.")
        return
    for title, m in reversed(notes):
        st.markdown(f'<div class="msg ai{" rtl" if URDU_RE.search(m["text"]) else ""}">'
                    f'<div class="who">{html.escape(title)} · {m["duration"]:.0f}s</div>'
                    f'{html.escape(m["text"]).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        render_player(m["audio"], m["id"], autoplay=False)


def page_settings(client) -> None:
    st.markdown('<div class="page-title">Settings & Help</div>'
                '<div class="page-sub">Connection, usage and conversation management.</div>', unsafe_allow_html=True)
    show_flash()

    st.markdown("### API connection")
    key = get_api_key()
    if not key:
        st.error("No API key found.")
        api_card()
    else:
        col1, col2 = st.columns([1, 2])
        if col1.button("🔌 Test connection"):
            try:
                with st.spinner("Contacting Gemini…"):
                    call_models(TEXT_MODELS, lambda m: client.models.generate_content(
                        model=m, contents="Reply with the single word OK."))
                st.session_state.api_status = "ok"
            except Exception as e:  # noqa: BLE001
                st.session_state.api_status = "error"
                st.error(friendly_error(e))
        status = st.session_state.api_status
        col2.markdown({"ok": '<span class="chip ok">● Connected</span>',
                       "error": '<span class="chip bad">● Connection problem</span>'}.get(
                           status, '<span class="chip">● Key found - not tested yet</span>'),
                      unsafe_allow_html=True)
        st.caption(f"Text model: {TEXT_MODELS[0]} · Speech models: {', '.join(TTS_MODELS)}")

    st.markdown("### Usage (this session)")
    s = st.session_state.stats
    a, b, c, d = st.columns(4)
    a.metric("AI requests", s["requests"])
    b.metric("Voice notes", s["voices"])
    c.metric("Audio generated", f'{s["audio_sec"]:.0f}s')
    d.metric("Errors", s["errors"])

    st.markdown("### Conversations")
    total = sum(len(c["messages"]) for c in st.session_state.conversations.values())
    st.caption(f"{len(st.session_state.conversations)} conversation(s), {total} message(s). "
               "History lives only in this browser session and disappears when you close or refresh the tab.")
    export = {cid: {"title": c["title"], "created": c["created"],
                    "messages": [{"role": m["role"], "text": m["text"]} for m in c["messages"]]}
              for cid, c in st.session_state.conversations.items()}
    st.download_button("⬇ Export history (JSON)", json.dumps(export, ensure_ascii=False, indent=2),
                       file_name="abdur-ai-history.json", mime="application/json")
    if st.button("🗑 Clear all history"):
        st.session_state.conversations = {}
        new_conversation()
        st.rerun()

    st.markdown("### Help")
    st.markdown(
        '<div class="card"><ul>'
        "<li><b>Microphone blocked?</b> Click the lock icon in the address bar and allow the microphone, then reload.</li>"
        "<li><b>No sound?</b> Check device volume. If autoplay is blocked, press ▶ on the voice note.</li>"
        "<li><b>Urdu:</b> speak Urdu or choose Urdu in Quick settings - answers appear in Urdu script.</li>"
        "<li><b>Privacy:</b> your API key stays on the server and is never sent to the browser.</li></ul></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(has_key: bool) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="dot"></div><div><div class="name">Abdur AI Assistant</div>'
                    '<div class="sub">Intelligent voice assistant</div></div></div>', unsafe_allow_html=True)
        st.radio("Navigate", ["🎙️ Voice Assistant", "💬 Text Chat", "🎧 Voice Notes", "⚙️ Settings & Help"],
                 key="page", label_visibility="collapsed")
        st.markdown('<span class="chip ok">● API key ready</span>' if has_key
                    else '<span class="chip bad">● API key missing</span>', unsafe_allow_html=True)

        with st.expander("🎚️ Quick settings", expanded=False):
            st.selectbox("Response language", LANGUAGES, key="lang")
            st.selectbox("Voice", VOICES, key="voice")
            st.select_slider("Speaking speed", options=[0.75, 0.9, 1.0, 1.15, 1.3, 1.5], value=1.0, key="voice_speed")
            st.selectbox("Assistant tone", list(STYLES), key="style")

        st.markdown("##### Conversations")
        if st.button("➕ New conversation", use_container_width=True, key="new_conv"):
            conv = current_conv()
            if conv["messages"]:
                new_conversation()
            st.rerun()

        for cid in reversed(list(st.session_state.conversations)):
            c = st.session_state.conversations[cid]
            active = cid == st.session_state.current
            c1, c2 = st.columns([5, 1])
            if c1.button(("● " if active else "") + c["title"], key=f"sel_{cid}", use_container_width=True):
                st.session_state.current = cid
                st.rerun()
            if c2.button("✕", key=f"del_{cid}", help="Delete conversation"):
                st.session_state.conversations.pop(cid, None)
                if not st.session_state.conversations:
                    new_conversation()
                elif st.session_state.current == cid:
                    st.session_state.current = list(st.session_state.conversations)[-1]
                st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    key = get_api_key()
    client = None
    if key:
        try:
            client = make_client(key)
        except Exception as e:  # noqa: BLE001
            st.session_state.flash = ("error", friendly_error(e))
    sidebar(bool(key))

    page = st.session_state.page
    if page.startswith("🎙️"):
        page_voice(client)
    elif page.startswith("💬"):
        page_text(client)
    elif page.startswith("🎧"):
        page_notes(client)
    else:
        page_settings(client)


main()
