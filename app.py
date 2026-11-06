"""
Abdur AI Assistant - an AI voice assistant built with Streamlit + Google Gemini.

Two voice engines (choose in the sidebar -> Quick settings):

1. Live voice-to-voice (default): your recording is sent straight to
   `gemini-3.8-live`, which listens and answers with its own voice in ONE call.
   Free tier available with a free Google AI Studio key.

2. Classic: Gemini understands the audio and writes the answer
   (`gemini-3.5-flash-lite`), then Gemini TTS (`gemini-3.8-flash-tts`) speaks it.

If Live fails for any reason, the app automatically falls back to Classic.

Setup
-----
    pip install -U streamlit google-genai
    export GEMINI_API_KEY="your-free-key"      # or put it in .streamlit/secrets.toml
    streamlit run app.py

Model names can be changed WITHOUT editing code. Set any of these in Streamlit
Secrets or as environment variables (comma separated lists are tried in order):

    TEXT_MODELS = "gemini-3.5-flash-lite,gemini-3.8-flash"
    TTS_MODELS  = "gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts"
    LIVE_MODEL  = "gemini-3.8-live"
"""

import asyncio
import base64
import html
import io
import json
import os
import re
import tempfile
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

# --------------------------------------------------------------------------- #
# Page config (must be the first Streamlit command)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Abdur AI Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def cfg_value(name: str, default: str = "") -> str:
    """Read a setting from Streamlit Secrets, then environment, then default."""
    try:
        v = st.secrets.get(name, "")
        if v:
            return str(v).strip()
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(name, "").strip() or default


def cfg_list(name: str, default: str) -> list:
    return [x.strip() for x in cfg_value(name, default).split(",") if x.strip()]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_NAME = "Abdur AI Assistant"

# Text / audio-understanding models (tried in order). All have a free tier.
TEXT_MODELS = cfg_list("TEXT_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash,gemini-3.6-flash")
# Text-to-speech models (tried in order). The old 2.5 / 3.1 preview TTS models
# are retired on 17 Nov 2026, so only the 3.8 models are listed.
TTS_MODELS = cfg_list("TTS_MODELS", "gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts")
# Live (audio-in -> audio-out) model.
LIVE_MODEL = cfg_value("LIVE_MODEL", "gemini-3.8-live")

ENGINE_LIVE = "Live voice-to-voice (recommended)"
ENGINE_CLASSIC = "Classic (understand + speak)"
ENGINES = [ENGINE_LIVE, ENGINE_CLASSIC]

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

MAX_TTS_CHARS = 2500    # characters spoken per answer (full text is always shown)
TTS_CHUNK_CHARS = 800   # each TTS request handles at most this many characters
MAX_HISTORY_MESSAGES = 24
LIVE_TIMEOUT_SEC = 70   # max time for one Live voice turn
LIVE_IN_RATE = 16000    # Live API input: 16-bit PCM, 16 kHz, mono
LIVE_OUT_RATE = 24000   # Live API output: 16-bit PCM, 24 kHz, mono

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
  border:1px solid rgba(168,85,247,.35); background:rgba(147,51,234,.12); margin-right:.3rem; }
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

.mic-hint{ text-align:center; color:var(--muted); font-size:.85rem; margin:.4rem 0 1.2rem 0; }
.engine-row{ text-align:center; margin:.2rem 0 .4rem 0; }

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
# Voice recorder component (explicit Start / Stop buttons, records in browser,
# converts to 16 kHz mono WAV and sends it to Python)
# --------------------------------------------------------------------------- #
RECORDER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
 html,body{margin:0;padding:0;background:transparent;font-family:Inter,Arial,sans-serif;color:#F8FAFC;}
 .wrap{display:flex;flex-direction:column;align-items:center;gap:10px;padding:6px 4px 10px 4px;}
 .row{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;}
 button{cursor:pointer;border:none;border-radius:999px;padding:14px 26px;font-size:16px;font-weight:600;color:#fff;
   transition:all .2s ease;min-width:170px;}
 #start{background:linear-gradient(135deg,#9333EA,#A855F7);box-shadow:0 0 24px rgba(168,85,247,.55);}
 #start:hover:not(:disabled){box-shadow:0 0 34px rgba(34,211,238,.6);transform:translateY(-1px);}
 #stop{background:linear-gradient(135deg,#dc2626,#f43f5e);box-shadow:0 0 24px rgba(244,63,94,.45);}
 button:disabled{opacity:.35;cursor:not-allowed;box-shadow:none;}
 .status{display:flex;align-items:center;gap:10px;min-height:28px;color:#94A3B8;font-size:14px;text-align:center;}
 .dot{width:12px;height:12px;border-radius:50%;background:#475569;}
 .rec .dot{background:#f43f5e;animation:pulse 1s ease-in-out infinite;}
 .rec .status,.rec{color:#fda4af;}
 .busy .dot{background:#22D3EE;animation:pulse .6s ease-in-out infinite;}
 @keyframes pulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.7);opacity:.35}}
 .bars{display:none;gap:3px;align-items:center;height:22px;}
 .rec .bars{display:flex;}
 .bars i{display:block;width:4px;background:#22D3EE;border-radius:2px;animation:bar 0.9s ease-in-out infinite;}
 .bars i:nth-child(2){animation-delay:.12s}.bars i:nth-child(3){animation-delay:.24s}
 .bars i:nth-child(4){animation-delay:.36s}.bars i:nth-child(5){animation-delay:.48s}
 @keyframes bar{0%,100%{height:5px}50%{height:22px}}
</style></head><body>
<div class="wrap" id="wrap">
  <div class="row">
    <button id="start">&#127897; Start recording</button>
    <button id="stop" disabled>&#9632; Stop &amp; send</button>
  </div>
  <div class="status"><span class="dot"></span><span class="bars"><i></i><i></i><i></i><i></i><i></i></span>
    <span id="msg">Press Start, speak, then press Stop &amp; send.</span></div>
</div>
<script>
function post(type,data){window.parent.postMessage(Object.assign({isStreamlitMessage:true,type:type},data||{}),"*");}
function setH(){post("streamlit:setFrameHeight",{height:document.documentElement.scrollHeight+4});}
post("streamlit:componentReady",{apiVersion:1});
const wrap=document.getElementById("wrap"),startB=document.getElementById("start"),
      stopB=document.getElementById("stop"),msgE=document.getElementById("msg");
const MAX_SECONDS=120;
let rec=null,stream=null,chunks=[],tick=null,t0=0;
function say(t){msgE.textContent=t;setH();}
function state(s){wrap.className="wrap "+s;startB.disabled=(s!=="");stopB.disabled=(s!=="rec");setH();}
function errText(e){
  const n=(e&&e.name)||"";
  if(n==="NotAllowedError"||n==="SecurityError")return "Microphone permission denied. Click the lock icon in the address bar, allow the microphone, then reload.";
  if(n==="NotFoundError"||n==="OverconstrainedError")return "No microphone was found. Connect one and try again.";
  if(n==="NotReadableError")return "The microphone is being used by another app. Close it and try again.";
  return "Could not start the microphone: "+(e&&e.message?e.message:n);
}
startB.onclick=async function(){
  if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia||!window.MediaRecorder){
    say("Recording is not supported here. Use Chrome, Edge, Firefox or Safari over HTTPS.");return;}
  try{stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});}
  catch(e){say(errText(e));return;}
  chunks=[];
  try{rec=new MediaRecorder(stream);}catch(e){say("Recording failed to start: "+e.message);return;}
  rec.ondataavailable=function(ev){if(ev.data&&ev.data.size)chunks.push(ev.data);};
  rec.onstop=onStopped;
  rec.start();t0=Date.now();state("rec");say("Recording… 0s - speak now");
  tick=setInterval(function(){
    const s=Math.floor((Date.now()-t0)/1000);say("Recording… "+s+"s - press Stop & send when done");
    if(s>=MAX_SECONDS&&rec&&rec.state!=="inactive")rec.stop();},500);
};
stopB.onclick=function(){if(rec&&rec.state!=="inactive")rec.stop();};
async function onStopped(){
  clearInterval(tick);
  if(stream){stream.getTracks().forEach(function(t){t.stop();});}
  const secs=(Date.now()-t0)/1000;
  state("busy");say("Processing your voice…");
  if(secs<0.6||!chunks.length){state("");say("That was too short. Press Start and speak for a moment.");return;}
  try{
    const blob=new Blob(chunks,{type:(rec&&rec.mimeType)||"audio/webm"});
    const buf=await blob.arrayBuffer();
    const AC=window.AudioContext||window.webkitAudioContext;
    const ctx=new AC();
    const decoded=await new Promise(function(res,rej){ctx.decodeAudioData(buf,res,rej);});
    try{ctx.close();}catch(e){}
    const wav=await toWav(decoded);
    const b64=await toB64(new Blob([wav],{type:"audio/wav"}));
    post("streamlit:setComponentValue",{value:{id:Date.now()+"-"+Math.random().toString(36).slice(2),
        mime:"audio/wav",b64:b64,seconds:secs},dataType:"json"});
    state("busy");say("Sent! The AI is answering…");
    setTimeout(function(){state("");say("Press Start to ask another question.");},2500);
  }catch(e){state("");say("Could not process the recording ("+(e&&e.message?e.message:"decode error")+"). Please try again.");}
}
async function toWav(audioBuf){
  const rate=16000,len=Math.max(1,Math.ceil(audioBuf.duration*rate));
  const off=new OfflineAudioContext(1,len,rate);
  const src=off.createBufferSource();src.buffer=audioBuf;src.connect(off.destination);src.start(0);
  const out=(await off.startRendering()).getChannelData(0);
  const pcm=new DataView(new ArrayBuffer(44+out.length*2));
  function str(o,s){for(let i=0;i<s.length;i++)pcm.setUint8(o+i,s.charCodeAt(i));}
  str(0,"RIFF");pcm.setUint32(4,36+out.length*2,true);str(8,"WAVE");str(12,"fmt ");
  pcm.setUint32(16,16,true);pcm.setUint16(20,1,true);pcm.setUint16(22,1,true);
  pcm.setUint32(24,rate,true);pcm.setUint32(28,rate*2,true);pcm.setUint16(32,2,true);pcm.setUint16(34,16,true);
  str(36,"data");pcm.setUint32(40,out.length*2,true);
  for(let i=0;i<out.length;i++){const v=Math.max(-1,Math.min(1,out[i]));pcm.setInt16(44+i*2,v<0?v*32768:v*32767,true);}
  return pcm.buffer;
}
function toB64(blob){return new Promise(function(res,rej){const r=new FileReader();
  r.onload=function(){res(String(r.result).split(",")[1]);};r.onerror=rej;r.readAsDataURL(blob);});}
window.addEventListener("message",function(){setH();});
setH();
</script></body></html>"""


@st.cache_resource(show_spinner=False)
def get_recorder():
    folder = Path(tempfile.gettempdir()) / "abdur_ai_recorder_v1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text(RECORDER_HTML, encoding="utf-8")
    return components.declare_component("abdur_voice_recorder", path=str(folder))


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
    ss.setdefault("last_rec_id", None)
    ss.setdefault("autoplay_id", None)
    ss.setdefault("flash", None)
    ss.setdefault("api_status", "unknown")
    ss.setdefault("user_key", "")
    ss.setdefault("engine", ENGINE_LIVE)
    ss.setdefault("stats", {"requests": 0, "voices": 0, "audio_sec": 0.0, "errors": 0, "fallbacks": 0})
    ss.stats.setdefault("fallbacks", 0)
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
    return cfg_value("GEMINI_API_KEY", "")


@st.cache_resource(show_spinner=False)
def make_client(key: str):
    # Force v1beta — speechConfig / TTS is still only available on the beta endpoint.
    # The SDK now defaults to stable v1, which rejects "speechConfig".
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=90_000,
            api_version="v1beta",
        ),
    )


class AppError(Exception):
    """Error with a message that is safe to show to the user."""


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if isinstance(e, AppError):
        return msg
    if isinstance(e, asyncio.TimeoutError):
        return "The live voice model took too long to answer."
    if "api_key_invalid" in low or "api key not valid" in low or "invalid api key" in low:
        return "Your Gemini API key is invalid. Check GEMINI_API_KEY and try again."
    if "permission_denied" in low or "403" in low:
        return "Gemini refused the request (403). The key may lack access to this model or region."
    if "429" in low or "resource_exhausted" in low or "quota" in low:
        return "Gemini rate limit or quota reached. Wait a minute and try again."
    if "404" in low or "no longer available" in low or "not found" in low:
        return ("A configured Gemini model is retired or unavailable (404). Update TEXT_MODELS, TTS_MODELS "
                "or LIVE_MODEL (see Settings & Help -> Check model names).")
    if any(x in low for x in ("timed out", "timeout", "connection", "getaddrinfo", "network", "unreachable")):
        return "Network problem while contacting Gemini. Check your connection and try again."
    if "503" in low or "unavailable" in low or "overloaded" in low or "500" in low:
        return "Gemini is busy right now. Please try again in a moment."
    if "safety" in low or "blocked" in low:
        return "Gemini could not answer that request because of its safety filters."
    return f"Gemini error: {msg[:300]}"


def is_transient(e: Exception) -> bool:
    if isinstance(e, asyncio.TimeoutError):
        return True
    low = str(e).lower()
    return any(x in low for x in ("503", "500", "unavailable", "overloaded", "timed out", "timeout",
                                  "connection", "reset", "network", "1011", "1006"))


def is_model_missing(e: Exception) -> bool:
    low = str(e).lower()
    return ("404" in low or "not found" in low or "no longer available" in low
            or "not supported" in low or "is not found" in low)


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


def lang_and_style_rules() -> list:
    lang = st.session_state.get("lang", "Auto-detect")
    style = STYLES.get(st.session_state.get("style", "Friendly"), STYLES["Friendly"])
    if lang == "Auto-detect":
        lang_rule = ("Reply in the same language the user speaks or writes (English or Urdu). "
                     "If the user speaks Roman Urdu, reply in Urdu.")
    else:
        lang_rule = f"Always reply in {lang}, whatever language the user uses."
    return [
        f"You are {APP_NAME}, a {style} AI assistant that talks with people like a real person.",
        lang_rule,
    ]


def history_text(conv: dict, n: int = 8) -> str:
    """Short plain-text recap of the last messages (used as context for Live)."""
    lines = []
    for m in conv["messages"][-n:]:
        t = m["text"].strip()
        if t:
            who = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{who}: {t[:500]}")
    return "\n".join(lines)


def build_system(mode: str, conv: dict = None) -> str:
    """mode: 'text' | 'classic_voice' | 'live'"""
    rules = lang_and_style_rules()
    if mode in ("text", "classic_voice"):
        rules += [
            "When replying in Urdu, use proper Urdu script (Arabic-based, اردو) ONLY. "
            "Never use Devanagari/Hindi script and never use Roman Urdu in your reply.",
            "Write plain text only: no markdown, no asterisks, no bullet symbols, no emojis, "
            "because your reply will be read aloud.",
            "Use the earlier conversation for context.",
        ]
    if mode == "classic_voice":
        rules.append("Keep replies natural and spoken-style, usually 1 to 4 sentences unless the user asks for detail.")
        rules.append(
            "The user's latest message is an audio recording. Transcribe exactly what they said in the "
            "language they spoke (Urdu in Urdu script), then answer. If the audio is silent or unintelligible, "
            "return an empty transcript and a short reply asking them to try again."
        )
        rules.append('Return JSON: {"transcript": ..., "reply": ..., "language": "English" or "Urdu"}.')
    elif mode == "live":
        rules += [
            "You are in a spoken voice conversation. Answer out loud, naturally, usually in 1 to 4 sentences "
            "unless the user asks for detail.",
            "Speak plain natural language: never read out symbols, markdown, URLs or emojis.",
            "If you reply in Urdu, speak proper Urdu.",
            "If the audio is silent or unintelligible, politely ask the user to try again.",
        ]
        recap = history_text(conv) if conv else ""
        if recap:
            rules.append("Earlier conversation (context only, do not repeat it):\n" + recap)
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
    # No thinking_config on purpose: the Gemini 3.x models use a different
    # thinking control than 2.5 and default to fast, low-latency behaviour.
    kwargs = dict(system_instruction=system, temperature=0.7)
    if json_out:
        kwargs.update(response_mime_type="application/json", response_schema=RESPONSE_SCHEMA)
    return types.GenerateContentConfig(**kwargs)


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    raise AppError("Gemini returned an unreadable answer. Please try again.")


# ---- Classic pipeline: understand audio -> text answer --------------------- #
def voice_turn(client, audio: bytes, mime: str, conv: dict) -> dict:
    contents = history_contents(conv)
    contents.append(types.Content(role="user", parts=[
        types.Part.from_bytes(data=audio, mime_type=mime),
        types.Part.from_text(text="Transcribe my voice message and reply."),
    ]))
    cfg = gen_config(build_system("classic_voice"), json_out=True)
    resp = call_models(TEXT_MODELS, lambda m: client.models.generate_content(model=m, contents=contents, config=cfg))
    data = parse_json(resp.text)
    return {
        "transcript": str(data.get("transcript", "")).strip(),
        "reply": str(data.get("reply", "")).strip(),
    }


def text_turn(client, conv: dict, user_text: str) -> str:
    contents = history_contents(conv)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
    cfg = gen_config(build_system("text"))
    resp = call_models(TEXT_MODELS, lambda m: client.models.generate_content(model=m, contents=contents, config=cfg))
    out = (resp.text or "").strip()
    if not out:
        raise AppError("Gemini returned an empty answer. Please rephrase and try again.")
    return out


# ---- Speech generation (TTS) ------------------------------------------------ #
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


def wav_to_pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes())


def voice_config(voice: str) -> types.SpeechConfig:
    return types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
        )
    )


def synth_chunk(client, model: str, text: str, voice: str):
    prompt = f"Say in a warm, natural, conversational tone:\n{text}"
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=voice_config(voice),
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    try:
        part = resp.candidates[0].content.parts[0]
        data = part.inline_data.data
        mime = part.inline_data.mime_type or ""
    except Exception:  # noqa: BLE001
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


# ---- Live voice-to-voice (audio in -> audio out in one call) --------------- #
async def _live_once(client, pcm: bytes, system: str, voice: str):
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system,
        speech_config=voice_config(voice),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    # Trailing silence helps the server detect that the speaker has finished.
    silence = b"\x00\x00" * int(LIVE_IN_RATE * 0.8)
    payload = pcm + silence
    step = LIVE_IN_RATE * 2 // 2  # 0.5 second chunks (16-bit mono)
    audio, user_parts, ai_parts = bytearray(), [], []

    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
        for i in range(0, len(payload), step):
            await session.send_realtime_input(
                audio=types.Blob(data=payload[i:i + step], mime_type=f"audio/pcm;rate={LIVE_IN_RATE}")
            )
        await session.send_realtime_input(audio_stream_end=True)

        async for msg in session.receive():
            sc = getattr(msg, "server_content", None)
            if not sc:
                continue
            if getattr(sc, "model_turn", None) and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        audio.extend(inline.data)
            it = getattr(sc, "input_transcription", None)
            if it and getattr(it, "text", None):
                user_parts.append(it.text)
            ot = getattr(sc, "output_transcription", None)
            if ot and getattr(ot, "text", None):
                ai_parts.append(ot.text)
            if getattr(sc, "turn_complete", False):
                break

    return bytes(audio), "".join(user_parts).strip(), "".join(ai_parts).strip()


def live_voice_turn(client, wav: bytes, conv: dict):
    """Returns (reply_wav, user_text, ai_text, duration_seconds)."""
    pcm = wav_to_pcm(wav)
    system = build_system("live", conv)
    voice = st.session_state.get("voice", "Kore")
    last = None
    for attempt in range(2):
        try:
            audio, user_text, ai_text = asyncio.run(
                asyncio.wait_for(_live_once(client, pcm, system, voice), timeout=LIVE_TIMEOUT_SEC)
            )
            if not audio:
                raise AppError("The live model returned no audio.")
            dur = len(audio) / (2 * LIVE_OUT_RATE)
            return (pcm_to_wav(audio, LIVE_OUT_RATE),
                    user_text or "🎙️ (voice message)",
                    ai_text or "🔊 (voice reply)",
                    dur)
        except Exception as e:  # noqa: BLE001
            last = e
            if is_transient(e) and attempt == 0:
                time.sleep(1.2)
                continue
            raise
    raise last if last else AppError("Live voice failed.")


def run_voice_turn(client, data: bytes, conv: dict, orb_ph) -> dict:
    """Run one voice turn with the chosen engine; falls back to Classic."""
    note = ""
    if st.session_state.get("engine", ENGINE_LIVE) == ENGINE_LIVE:
        orb_ph.markdown(orb("thinking", "Listening and thinking…"), unsafe_allow_html=True)
        try:
            wav, user_text, ai_text, dur = live_voice_turn(client, data, conv)
            return {"transcript": user_text, "reply": ai_text, "wav": wav, "duration": dur, "note": ""}
        except Exception as e:  # noqa: BLE001
            st.session_state.stats["errors"] += 1
            st.session_state.stats["fallbacks"] += 1
            note = f"Live voice mode failed ({friendly_error(e)}) so I switched to classic mode for this answer."

    orb_ph.markdown(orb("thinking", "Listening to your words…"), unsafe_allow_html=True)
    result = voice_turn(client, data, "audio/wav", conv)
    return {"transcript": result["transcript"], "reply": result["reply"],
            "wav": None, "duration": 0.0, "note": note}


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
    """Generate speech for a message with the TTS models. Returns '' or an error string."""
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
        if m["role"] not in ("ai", "assistant"):
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
        "You can get a free key at Google AI Studio. You can also paste a key below for this browser "
        "session only - it is never saved.</div>",
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
        "<li>Press <b>Start recording</b> and allow microphone access when your browser asks.</li>"
        "<li>Speak naturally in English or Urdu, then press <b>Stop &amp; send</b>.</li>"
        "<li>I listen and answer with my own voice, automatically.</li>"
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

    live_on = st.session_state.get("engine", ENGINE_LIVE) == ENGINE_LIVE
    speaking = st.session_state.autoplay_id is not None
    orb_ph = st.empty()
    orb_ph.markdown(orb("speaking", "Speaking…") if speaking else orb("idle", "Press Start recording and talk"),
                    unsafe_allow_html=True)
    st.markdown(f'<div class="engine-row"><span class="chip ok">{"⚡ Live voice-to-voice" if live_on else "🔁 Classic mode"}'
                f' · {html.escape(LIVE_MODEL if live_on else TEXT_MODELS[0])}</span></div>',
                unsafe_allow_html=True)

    recorder = get_recorder()
    rec_value = recorder(key="voice_rec", default=None)
    st.markdown('<div class="mic-hint">Press <b>Start recording</b>, speak, then press <b>Stop &amp; send</b>. '
                "I answer out loud automatically. Your browser will ask for microphone permission the first time.</div>",
                unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("⏹ Stop speaking", key="stop_speaking"):
        stop_all_audio()
    if c2.button("➕ New chat", key="new_chat_voice"):
        new_conversation()
        st.rerun()

    if isinstance(rec_value, dict) and rec_value.get("id") and rec_value["id"] != st.session_state.last_rec_id:
        st.session_state.last_rec_id = rec_value["id"]
        try:
            data = base64.b64decode(rec_value.get("b64", ""))
        except Exception:  # noqa: BLE001
            data = b""
        if len(data) < 3000:
            st.session_state.flash = ("warning", "I did not hear anything. Please record again and speak for a moment.")
            st.rerun()
        try:
            st.session_state.stats["requests"] += 1
            result = run_voice_turn(client, data, conv, orb_ph)
            if not result["transcript"]:
                st.session_state.flash = ("warning", "I could not understand the recording. Please try again, a little closer to the microphone.")
                st.rerun()

            add_message(conv, "user", result["transcript"])
            reply_text = result["reply"] or "Sorry, I could not think of an answer."
            ai_msg = add_message(conv, "ai", reply_text, audio=result["wav"], duration=result["duration"])

            if result["wav"]:  # Live engine already produced the voice
                st.session_state.stats["voices"] += 1
                st.session_state.stats["audio_sec"] += result["duration"]
                st.session_state.autoplay_id = ai_msg["id"]
            else:              # Classic engine: speak the text answer with TTS
                orb_ph.markdown(orb("thinking", "Creating my voice…"), unsafe_allow_html=True)
                err = attach_speech(client, conv, ai_msg)
                if err:
                    st.session_state.flash = ("error", f"I answered in text, but voice generation failed: {err}")

            if result["note"] and not st.session_state.flash:
                st.session_state.flash = ("warning", result["note"])
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


def available_model_ids(client) -> set:
    names = set()
    for m in client.models.list():
        n = getattr(m, "name", "") or ""
        names.add(n.replace("models/", ""))
    return names


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
        col1, col2, col3 = st.columns([1, 1, 1])
        if col1.button("🔌 Test text model"):
            try:
                with st.spinner("Contacting Gemini…"):
                    call_models(TEXT_MODELS, lambda m: client.models.generate_content(
                        model=m, contents="Reply with the single word OK."))
                st.session_state.api_status = "ok"
                st.success("Text model works.")
            except Exception as e:  # noqa: BLE001
                st.session_state.api_status = "error"
                st.error(friendly_error(e))
        if col2.button("🔊 Test speech (TTS)"):
            try:
                with st.spinner("Generating a short voice sample…"):
                    wav, dur = generate_speech(client, "Hello! This is a test of my voice.")
                st.success(f"Speech works ({dur:.1f}s).")
                render_player(wav, "test", autoplay=False)
            except Exception as e:  # noqa: BLE001
                st.error(friendly_error(e))
        if col3.button("🧾 Check model names"):
            try:
                with st.spinner("Asking Gemini which models exist…"):
                    ids = available_model_ids(client)
                rows = [("Text", m) for m in TEXT_MODELS] + [("Speech", m) for m in TTS_MODELS] + [("Live", LIVE_MODEL)]
                for kind, m in rows:
                    st.write(f'{"✅" if m in ids else "❌"} **{kind}**: `{m}`'
                             + ("" if m in ids else " - not listed for your key (may still work, or update the name)"))
            except Exception as e:  # noqa: BLE001
                st.error(friendly_error(e))

        status = st.session_state.api_status
        st.markdown({"ok": '<span class="chip ok">● Connected</span>',
                     "error": '<span class="chip bad">● Connection problem</span>'}.get(
                         status, '<span class="chip">● Key found - not tested yet</span>'),
                    unsafe_allow_html=True)
        st.caption(f"Live model: {LIVE_MODEL} · Text models: {', '.join(TEXT_MODELS)} · "
                   f"Speech models: {', '.join(TTS_MODELS)}")

    st.markdown("### Usage (this session)")
    s = st.session_state.stats
    a, b, c, d, e = st.columns(5)
    a.metric("AI requests", s["requests"])
    b.metric("Voice notes", s["voices"])
    c.metric("Audio generated", f'{s["audio_sec"]:.0f}s')
    d.metric("Errors", s["errors"])
    e.metric("Live fallbacks", s.get("fallbacks", 0))

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
        "<li><b>Urdu:</b> speak Urdu or choose Urdu in Quick settings - Classic mode shows answers in Urdu script.</li>"
        "<li><b>Rate limit (429)?</b> The free tier has per-project limits. Wait a minute, or switch the voice "
        "engine to Classic in Quick settings.</li>"
        "<li><b>404 / model retired?</b> Press <i>Check model names</i> above, then set TEXT_MODELS, TTS_MODELS "
        "or LIVE_MODEL in Streamlit Secrets.</li>"
        "<li><b>Privacy:</b> your API key stays on the server and is never sent to the browser. On Google's free "
        "tier, content you send may be used to improve Google products, so avoid private audio.</li></ul></div>",
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
            st.radio("Voice engine", ENGINES, key="engine",
                     help="Live = one model listens and speaks (fastest). "
                          "Classic = understand the audio, then read the answer with text-to-speech.")
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
