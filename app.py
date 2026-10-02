"""Beyond Vision - Multimodal Threat Engine (Streamlit).

Key design points
-----------------
* Model IDs are never trusted blindly: the app asks each provider which models
  exist, ranks them, and walks a fallback chain, so a retired model can no
  longer take the whole feature down.
* Override the models any time with GEMINI_MODEL_ID / GROQ_MODEL_ID
  (Streamlit secret or environment variable). No code change needed.
* Every piece of user / LLM text is escaped before it touches raw HTML.
"""
from __future__ import annotations

import hashlib
import html as html_lib
import io
import os
import re
from urllib.parse import urlparse

import streamlit as st
from PIL import Image, ImageOps

# ------------------------------------------------------------------
# Optional dependencies (the app degrades gracefully without them)
# ------------------------------------------------------------------
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import pytesseract
    try:
        pytesseract.get_tesseract_version()  # the binary must exist, not just the wrapper
        PYTESSERACT_AVAILABLE = True
    except Exception:
        PYTESSERACT_AVAILABLE = False
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ==========================================
# MODEL CONFIGURATION
# ==========================================
# Last-resort lists, used only if live model discovery fails.
# Newest first. Add/remove freely; discovery makes this self-correcting.
GEMINI_FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
GROQ_FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
GROQ_EXCLUDE = ("whisper", "guard", "tts", "orpheus", "playai", "embed", "safeguard")
MAX_ATTEMPTS = 5            # max models tried per request
MAX_INLINE_MB = 18          # Gemini inline request limit is ~20 MB total
MAX_TEXT_CHARS = 6000       # text sent to the LLMs

MEDIA_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".mp4": "video/mp4",
}

st.set_page_config(page_title="Beyond Vision", page_icon="☁️", layout="centered")

# ==========================================
# STYLE - Dark radial canvas & gold accents
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,500&family=Inter:wght@300;400;500;600&display=swap');
header, .stAppHeader, #MainMenu, footer, .stAppDeployButton,
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stToolbar"], div[class*="viewerBadge"] { display: none !important; visibility: hidden !important; }
.block-container { padding-top: 1rem !important; }
:root {
    --gold: #D4AF37; --cream: #EAE3CB; --cream-bright: #F7F5EB; --muted: #ABABA4;
    --safe: #8BB98F; --danger: #C3676B; --warning: #E0A96D; --unknown: #8C9AA6;
    --plate-bg: rgba(255,255,255,0.03); --plate-border: rgba(255,255,255,0.08);
}
.stApp { background: radial-gradient(circle at 50% 0%, #2A2D24 0%, #0D0E0B 70%) !important; font-family: 'Inter', sans-serif; }
h1, h2, h3, .serif-header { font-family: 'Playfair Display', serif !important; color: var(--cream) !important; font-weight: 500; letter-spacing: .3px; }
p, li, label, span { font-family: 'Inter', sans-serif; }
::-webkit-scrollbar { height: 6px; width: 6px; }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,.4); border-radius: 10px; }
::-webkit-scrollbar-track { background: transparent; }
div[data-testid="stRadio"] > div[role="radiogroup"] { display: flex; flex-wrap: wrap; gap: 10px; }
div[data-testid="stRadio"] label { background: var(--plate-bg); border: 1px solid var(--plate-border); border-radius: 30px; padding: 10px 20px !important; transition: all .2s ease; cursor: pointer; }
div[data-testid="stRadio"] label:hover { border-color: rgba(212,175,55,.5); }
div[data-testid="stRadio"] label > div:first-child { display: none !important; }
div[data-testid="stRadio"] label:has(input:checked) { background: linear-gradient(135deg, var(--gold) 0%, #B8912B 100%); border-color: var(--gold); }
div[data-testid="stRadio"] label:has(input:checked) p { color: #16170F !important; font-weight: 500; }
div[data-testid="stRadio"] label p { color: var(--cream) !important; font-size: .92rem; margin: 0; }
.stTextArea textarea { background: rgba(255,255,255,.03) !important; color: var(--cream) !important; border: 1px solid var(--plate-border) !important; border-radius: 14px !important; }
.stTextArea textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 1px rgba(212,175,55,.4) !important; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.02) !important; border: 1px dashed var(--plate-border) !important; border-radius: 16px !important; }
[data-testid="stFileUploaderDropzone"] button { background: rgba(212,175,55,.12) !important; color: var(--cream) !important; border: 1px solid rgba(212,175,55,.3) !important; }
.stButton button { background: linear-gradient(135deg, var(--gold) 0%, #B8912B 100%) !important; color: #16170F !important; border: none !important; border-radius: 30px !important; font-weight: 500 !important; padding: 10px 26px !important; transition: transform .15s ease, box-shadow .15s ease !important; }
.stButton button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(212,175,55,.25); }
.stButton button:disabled { opacity: .45; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 24px !important; background-color: rgba(255,255,255,.02) !important; border: 1px solid rgba(255,255,255,.06) !important; box-shadow: 0 10px 30px rgba(0,0,0,.4) !important; padding: 15px !important; }
.verdict-badge { display: inline-flex; flex-direction: column; align-items: flex-start; gap: 4px; padding: 14px 22px; border-radius: 14px; font-weight: 500; margin-bottom: 16px; border: 1px solid transparent; width: 100%; box-sizing: border-box; }
.verdict-spam { background: rgba(195,103,107,.12); border-color: rgba(195,103,107,.35); color: var(--danger); }
.verdict-safe { background: rgba(139,185,143,.12); border-color: rgba(139,185,143,.35); color: var(--safe); }
.verdict-unknown { background: rgba(224,169,109,.12); border-color: rgba(224,169,109,.35); color: var(--warning); }
.confidence-wrap { margin: 4px 0 20px 0; }
.confidence-track { width: 100%; height: 8px; border-radius: 8px; background: rgba(255,255,255,.06); overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 8px; }
.confidence-caption { margin-top: 6px; font-size: .82rem; color: var(--muted); }
.flag-list { color: var(--cream); font-size: .92rem; line-height: 1.7; padding-left: 1.2rem; }
.link-chip { display: inline-block; background: rgba(255,255,255,.03); border: 1px solid var(--plate-border); border-radius: 20px; padding: 6px 14px; margin: 4px 6px 4px 0; font-size: .82rem; color: var(--gold); font-family: monospace; word-break: break-all; }
.slider-container { display: flex; overflow-x: auto; gap: 20px; padding: 10px 0 20px 0; scroll-snap-type: x mandatory; scrollbar-width: none; }
.slider-container::-webkit-scrollbar { display: none; }
.dark-plate { min-width: min(320px, 88vw); max-width: 480px; flex: 0 0 85%; scroll-snap-align: center; background-color: var(--plate-bg); border-radius: 24px; padding: 35px; box-shadow: 0 8px 32px rgba(0,0,0,.3); border: 1px solid var(--plate-border); backdrop-filter: blur(10px); }
.dark-plate h3 { margin-top: 0; color: var(--gold) !important; font-size: 1.5rem; margin-bottom: 20px; }
.dark-plate p { font-size: .95rem; line-height: 1.7; color: var(--muted); font-weight: 300; }
.dark-plate b { color: var(--cream); font-weight: 500; }
.slider-hint { text-align: center; color: var(--muted); font-size: .8rem; letter-spacing: 1px; margin: -8px 0 10px 0; }
.footer { text-align: center; margin-top: 70px; padding-top: 30px; border-top: 1px solid rgba(255,255,255,.05); margin-bottom: 30px; }
.footer-title { font-size: .7rem; font-weight: 500; color: #8C8C87; margin-bottom: 15px; letter-spacing: 3px; text-transform: uppercase; }
.footer-link { display: inline-block; background-color: rgba(255,255,255,.03); color: var(--cream) !important; border: 1px solid rgba(255,255,255,.1); padding: 12px 28px; border-radius: 30px; text-decoration: none; font-size: .9rem; transition: .3s; }
.footer-link:hover { background-color: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.2); }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SMALL HELPERS
# ==========================================
def esc(value) -> str:
    """HTML-escape anything before it is placed in raw HTML."""
    return html_lib.escape(str(value), quote=True)


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def show_image(data, caption: str | None = None) -> None:
    """Version-proof st.image (use_container_width was replaced by width='stretch')."""
    try:
        st.image(data, caption=caption, width="stretch")
    except Exception:
        st.image(data, caption=caption, use_container_width=True)


def clean_markdown(text: str) -> str:
    """Strip markdown images from model output so nothing remote is auto-loaded."""
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text or "")


def defang(url: str) -> str:
    """Make a URL non-clickable so users never tap a live scam link."""
    return url.replace("http", "hxxp", 1).replace(".", "[.]")


# ==========================================
# RESOURCE INITIALIZATION
# ==========================================
@st.cache_resource(show_spinner="Loading threat classifier...")
def load_classifier():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        return pipeline(
            "text-classification",
            model="roshana1s/spam-message-classifier",
            truncation=True,
            max_length=512,
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_genai_client(api_key: str | None):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str | None):
    if not GROQ_AVAILABLE or not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def run_ocr(image_bytes: bytes) -> str:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    return pytesseract.image_to_string(image).strip()


ai_classifier = load_classifier()
genai_client = get_genai_client(get_setting("GEMINI_API_KEY"))
groq_client = get_groq_client(get_setting("GROQ_API_KEY"))


# ==========================================
# LLM LAYER - self-healing model selection
# ==========================================
class AllEnginesDown(Exception):
    """Raised when every configured engine failed. Carries per-model errors."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


_GEMINI_NAME = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?(-preview.*)?$")


@st.cache_resource(ttl=3600, show_spinner=False)
def discover_gemini_models(_client) -> list[str]:
    """Return generateContent-capable Gemini Flash models, best first. [] on failure."""
    ranked = []
    try:
        for m in _client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            hit = _GEMINI_NAME.match(name)
            if hit and ("generateContent" in actions or not actions):
                version = float(hit.group(1))
                is_lite = 1 if hit.group(2) else 0
                is_preview = 1 if hit.group(3) else 0
                ranked.append(((-version, is_lite, is_preview), name))
    except Exception:
        return []
    return [name for _, name in sorted(ranked)]


def gemini_model_chain(client) -> list[str]:
    discovered = discover_gemini_models(client)
    chain: list[str] = []
    override = get_setting("GEMINI_MODEL_ID")
    if override:
        chain.append(override)
    chain += discovered[:3]
    chain += [m for m in GEMINI_FALLBACK_MODELS if not discovered or m in discovered or m.endswith("-latest")]
    return list(dict.fromkeys(chain))[:MAX_ATTEMPTS]


@st.cache_resource(ttl=3600, show_spinner=False)
def discover_groq_models(_client) -> list[str]:
    try:
        ids = [m.id for m in _client.models.list().data]
    except Exception:
        return []
    return [i for i in ids if not any(x in i.lower() for x in GROQ_EXCLUDE)]


def groq_model_chain(client) -> list[str]:
    discovered = discover_groq_models(client)
    chain: list[str] = []
    override = get_setting("GROQ_MODEL_ID")
    if override:
        chain.append(override)
    chain += [m for m in GROQ_FALLBACK_MODELS if not discovered or m in discovered]
    chain += [m for m in discovered if m not in chain]
    return list(dict.fromkeys(chain))[:MAX_ATTEMPTS]


def _is_auth_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()
    return code in (401, 403) or "api key not valid" in text or "invalid api key" in text


def gemini_generate(contents) -> tuple[str, str]:
    """Try Gemini models in order. Returns (text, model_used)."""
    if genai_client is None:
        raise AllEnginesDown(["Gemini: no API key or SDK unavailable."])
    errors: list[str] = []
    for model in gemini_model_chain(genai_client):
        try:
            response = genai_client.models.generate_content(model=model, contents=contents)
            text = (getattr(response, "text", None) or "").strip()
            if text:
                return text, model
            errors.append(f"{model}: empty response (possibly blocked by safety filters)")
        except Exception as exc:
            errors.append(f"{model}: {str(exc)[:220]}")
            if _is_auth_error(exc):
                break
    raise AllEnginesDown(errors)


def groq_generate(prompt: str) -> tuple[str, str]:
    """Text-only fallback. Returns (text, model_used)."""
    if groq_client is None:
        raise AllEnginesDown(["Groq: no API key or SDK unavailable."])
    errors: list[str] = []
    for model in groq_model_chain(groq_client):
        try:
            completion = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            text = (completion.choices[0].message.content or "").strip()
            if text:
                return text, model
            errors.append(f"{model}: empty response")
        except Exception as exc:
            errors.append(f"{model}: {str(exc)[:220]}")
            if _is_auth_error(exc):
                break
    raise AllEnginesDown(errors)


def analyze_text_with_llm(prompt: str) -> tuple[str, list[str]]:
    """Gemini first, Groq second. Returns (report_markdown, error_log)."""
    log: list[str] = []
    try:
        text, model = gemini_generate(prompt)
        return text, log
    except AllEnginesDown as exc:
        log += exc.errors
    try:
        text, model = groq_generate(prompt)
        return f"*Backup engine in use ({model})*\n\n{text}", log
    except AllEnginesDown as exc:
        log += exc.errors
    return "", log


# ==========================================
# THREAT ANALYSIS ENGINE
# ==========================================
KEYWORDS = [
    "urgent", "locked", "verify", "suspended", "won", "prize", "tax", "pan", "bonus",
    "deposit", "kyc", "wa.link", "claim", "lottery", "otp", "refund", "winner", "blocked",
    "expired", "final notice", "click here", "act now", "gift card",
]
KEYWORD_PATTERNS = [(k, re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)", re.I)) for k in KEYWORDS]
OTP_REQUEST = re.compile(r"\b(?:share|send|tell|give|enter|forward)\b[^.\n]{0,25}\b(?:otp|pin|cvv|password)\b", re.I)

SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "wa.link", "shorturl.at", "ow.ly", "tiny.cc")
SUSPICIOUS_TLDS = (".xyz", ".top", ".click", ".icu", ".tk", ".ml", ".ga", ".cf", ".gq", ".work", ".loan", ".zip", ".mov", ".buzz", ".rest", ".cam", ".monster", ".cfd", ".sbs")
BRANDS = ("paypal", "amazon", "google", "apple", "microsoft", "netflix", "sbi", "hdfc", "icici", "paytm", "fedex", "dhl", "whatsapp", "facebook", "instagram")
SLD_PREFIXES = {"co", "com", "org", "net", "gov", "ac", "edu"}

URL_RE = re.compile(
    r"(?:https?://|www\.|(?:" + "|".join(re.escape(s) for s in SHORTENERS) + r")/)[^\s<>\"'`]+",
    re.I,
)


def extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for raw in URL_RE.findall(text):
        url = raw.rstrip(".,;:!?)]}'\"")
        if url and url not in urls:
            urls.append(url)
    return urls


def analyze_url(url: str) -> tuple[int, list[str]]:
    flags: list[str] = []
    points = 0
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return 15, [f"Malformed link: {defang(url)}"]
    if not host:
        return 0, flags

    labels = host.split(".")
    registered = labels[-2] if len(labels) >= 2 else host
    if len(labels) >= 3 and labels[-2] in SLD_PREFIXES:
        registered = labels[-3]

    if host in SHORTENERS:
        points += 12; flags.append(f"URL shortener hides the real destination ({host}).")
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        points += 20; flags.append("Link uses a raw IP address instead of a domain name.")
    if any(host.endswith(t) for t in SUSPICIOUS_TLDS):
        points += 15; flags.append(f"Domain uses a TLD heavily abused by scammers ({labels[-1]}).")
    if "xn--" in host:
        points += 15; flags.append("Domain uses punycode (possible look-alike characters).")
    if "@" in (parsed.netloc or ""):
        points += 15; flags.append("Link contains '@', a classic trick to disguise the real host.")
    if len(labels) >= 5 or host.count("-") >= 3:
        points += 8; flags.append("Domain has an unusually long or hyphen-heavy structure.")
    if url.lower().startswith("http://"):
        points += 5; flags.append("Link is not encrypted (http, not https).")
    for brand in BRANDS:
        if brand in host and registered != brand:
            points += 20; flags.append(f"Domain mentions '{brand}' but is not the official {brand} domain.")
            break
    return points, flags


def heuristic_analysis(text: str) -> dict:
    risk = 15
    flags: list[str] = []

    matched = [k for k, pat in KEYWORD_PATTERNS if pat.search(text)]
    if matched:
        risk += min(35, 15 + 5 * len(matched))
        flags.append(f"Triggered high-risk social-engineering keywords: {esc(', '.join(matched))}.")

    if OTP_REQUEST.search(text):
        risk += 35
        flags.append("Language suggests a request for sensitive authentication data (e.g., OTP or PIN).")

    urls = extract_urls(text)
    for u in urls:
        u_pts, u_flags = analyze_url(u)
        risk += u_pts
        flags.extend(u_flags)

    if urls and not flags:
        flags.append("Contains links or domains (use caution).")
    elif not flags:
        flags.append("No overt structural heuristic red flags detected.")

    return {
        "score": min(risk, 100),
        "flags": flags,
        "urls": urls
    }


# ==========================================
# HEADER
# ==========================================
st.markdown("""
    <div style="text-align: center; margin-bottom: 35px; margin-top: 20px;">
        <p style="font-size: 0.75rem; letter-spacing: 3px; color: #D4AF37; text-transform: uppercase; margin-bottom: 5px; font-weight: 600;">Multimodal Threat Engine</p>
        <h1 style="font-size: 3.5rem; font-weight: 400; color: #F7F5EB; margin-top: 0; margin-bottom: 0;">Beyond <span style="color: #D4AF37; font-style: italic;">Vision</span></h1>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
for key, default in {
    "scan_result": None,
    "extracted_text": "",
    "llm_report": None,
    "deepfake_report": None,
    "last_media_hash": None,
    "last_input": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

nav = st.radio("Navigation", ["🔍 Threat Scanner", "📚 Cyber Awareness Guide"], horizontal=True, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# PAGE 1: THREAT SCANNER
# ==========================================
if nav == "🔍 Threat Scanner":
    if not TRANSFORMERS_AVAILABLE or ai_classifier is None:
        st.info("The text-classification model is offline. Media scanning is unaffected.")

    with st.container(border=True):
        st.markdown("<h3 class='serif-header' style='margin-bottom: 15px; font-size: 1.4rem; color: #D4AF37 !important;'>📥 Threat Ingestion Source</h3>", unsafe_allow_html=True)

        input_type = st.radio(
            "Select the type of threat to analyze:",
            ["Text Input (Link/Message)", "Image Upload (Screenshot OCR)", "Media Upload (Deepfake/Voice/Video)"],
            horizontal=False,
            label_visibility="collapsed"
        )

        user_input = ""

        # Option A: OCR Upload
        if input_type == "Image Upload (Screenshot OCR)":
            uploaded_file = st.file_uploader("Upload a screenshot of the SMS, WhatsApp, or Email", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                show_image(image_bytes, caption="Uploaded Evidence")

                if not PYTESSERACT_AVAILABLE:
                    st.error("OCR is unavailable on this server. Please paste the text manually below.")
                else:
                    with st.spinner("Extracting text via Optical Character Recognition..."):
                        try:
                            st.session_state["extracted_text"] = run_ocr(image_bytes)
                        except Exception as e:
                            st.session_state["extracted_text"] = ""
                            st.error(f"OCR failed: {esc(e)}")

            user_input = st.text_area("Extracted Text (Edit if necessary):", value=st.session_state["extracted_text"], height=150)

        # Option B: Deepfake Forensics Upload
        elif input_type == "Media Upload (Deepfake/Voice/Video)":
            uploaded_media = st.file_uploader("Upload Audio (mp3, wav), Video (mp4), or Photo (jpg, png)", type=["png", "jpg", "jpeg", "mp3", "wav", "mp4"])

            if uploaded_media is not None:
                media_bytes = uploaded_media.getvalue()
                current_hash = hashlib.sha256(media_bytes).hexdigest()
                
                if st.session_state["last_media_hash"] != current_hash:
                    st.session_state["deepfake_report"] = None
                    st.session_state["last_media_hash"] = current_hash

                file_name = uploaded_media.name.lower()
                _, ext = os.path.splitext(file_name)
                media_type = MEDIA_MIME.get(ext)

                if ext in (".png", ".jpg", ".jpeg"):
                    show_image(media_bytes)
                elif ext in (".mp3", ".wav"):
                    st.audio(media_bytes)
                elif ext == ".mp4":
                    st.video(media_bytes)

                size_mb = len(media_bytes) / (1024 * 1024)
                if size_mb > MAX_INLINE_MB:
                    st.warning(f"File size is {size_mb:.1f} MB. Inline files over {MAX_INLINE_MB} MB may be rejected by the API.")
                
                if not media_type:
                    st.error(f"Unsupported file extension: {esc(ext)}")
                else:
                    scan_disabled = genai_client is None
                    if scan_disabled:
                        st.info("Deepfake scanning requires a Gemini API key. Add GEMINI_API_KEY in Streamlit Secrets.")

                    if st.button("🔍 Run Deepfake Scan", disabled=scan_disabled):
                        with st.spinner("Analyzing media for AI generation artifacts..."):
                            prompt = "You are a senior forensic analyst for 'Beyond Vision'. Analyze the user's intent and context for indicators of AI generation, synthetic manipulation, or deepfake/voice-cloning artifacts. Provide: 1. Synthetic Probability Verdict (Real vs. AI-Generated) 2. Confidence Score (0-100%) 3. Key Forensic Observations 4. Potential Scam Context"
                            
                            report = None
                            try:
                                media_part = types.Part.from_bytes(data=media_bytes, mime_type=media_type)
                                report, model = gemini_generate([prompt, media_part])
                            except Exception as e:
                                st.toast("Gemini quota/servers exhausted. Rerouting to Groq backup...", icon="🔄")
                                try:
                                    fallback = groq_client.chat.completions.create(
                                        messages=[{"role": "user", "content": prompt + "\n[Note: Media file was uploaded, but primary vision engine is rate-limited. Provide general forensic indicators for this file type]."}],
                                        model="llama3-8b-8192"
                                    )
                                    report = f"**[Groq Text-Fallback Analysis]**\n\n{fallback.choices[0].message.content}"
                                except Exception as fallback_error:
                                    report = f"⚠️ All engines are fully maxed out.\nGemini Error: {e}\nGroq Error: {fallback_error}"

                            st.session_state["deepfake_report"] = clean_markdown(report)

            if st.session_state["deepfake_report"]:
                st.markdown("<h3 class='serif-header' style='margin-top: 25px; color: #D4AF37 !important;'>🤖 Forensics Report</h3>", unsafe_allow_html=True)
                st.info(st.session_state["deepfake_report"])

        # Option C: Direct Text Ingestion
        else:
            user_input = st.text_area("Paste suspicious message or URL here:", value=st.session_state.get("last_input", ""), height=150)

        # Core Text/OCR Execution Logic
        if input_type in ["Text Input (Link/Message)", "Image Upload (Screenshot OCR)"]:
            scan_clicked = st.button("Scan Threat Vectors", disabled=(ai_classifier is None and genai_client is None and groq_client is None))

            if scan_clicked:
                clean_text = user_input.strip()[:MAX_TEXT_CHARS]
                if not clean_text:
                    st.warning("Enter or extract some text before scanning.")
                else:
                    with st.spinner("Analyzing threat vectors..."):
                        st.session_state["last_input"] = clean_text
                        st.session_state["llm_report"] = None

                        # 1. ML Classifier
                        ai_label = "UNKNOWN"
                        ai_score = 0.0
                        if ai_classifier:
                            try:
                                ai_output = ai_classifier(clean_text)[0]
                                ai_label = ai_output.get("label", "UNKNOWN").upper()
                                ai_score = round(ai_output.get("score", 0.0) * 100, 2)
                            except Exception:
                                ai_label = "ERROR"

                        # 2. Original Heuristic & TLD Engine
                        h_res = heuristic_analysis(clean_text)
                        risk_score = h_res["score"]
                        
                        if ai_label in ["SPAM", "LABEL_1", "1"]:
                            risk_score = max(risk_score, int(ai_score * 0.85))

                        st.session_state["scan_result"] = {
                            "ai_label": ai_label,
                            "ai_score": ai_score,
                            "risk_score": min(risk_score, 100),
                            "heuristics": h_res["flags"],
                            "urls": h_res["urls"],
                        }

        # Results Display
        if st.session_state.get("scan_result"):
            res = st.session_state["scan_result"]

            st.markdown("<h3 class='serif-header' style='margin-top: 25px; color: #F7F5EB !important;'>🧠 Machine Learning Analysis</h3>", unsafe_allow_html=True)
            if res["ai_label"] in ["SPAM", "LABEL_1", "1"]:
                st.error(f"**Verdict: SPAM** (Confidence: {res['ai_score']}%)")
            elif res["ai_label"] != "UNKNOWN" and res["ai_label"] != "ERROR":
                st.success(f"**Verdict: SAFE** (Confidence: {res['ai_score']}%)")
            else:
                 st.info(f"Verdict: {res['ai_label']}")

            st.markdown("<h3 class='serif-header' style='margin-top: 25px; color: #F7F5EB !important;'>⚙️ Heuristic Analysis</h3>", unsafe_allow_html=True)
            
            if res["risk_score"] > 75:
                badge_class = "verdict-spam"
                badge_title = "🚨 HIGH RISK"
            elif res["risk_score"] > 50:
                badge_class = "verdict-unknown"
                badge_title = "⚠️ MEDIUM RISK"
            else:
                badge_class = "verdict-safe"
                badge_title = "✅ LOW RISK"
                
            st.markdown(f"""
            <div class="verdict-badge {badge_class}">
                <div>{badge_title} (Score: {res['risk_score']}/100)</div>
            </div>
            """, unsafe_allow_html=True)

            if res.get("heuristics"):
                st.markdown(f"<p style='color: var(--cream); font-size: 0.95rem; margin-top: 10px;'><b>Breakdown:</b></p><ul class='flag-list'><li>{'</li><li>'.join(esc(f) for f in res['heuristics'])}</li></ul>", unsafe_allow_html=True)

            if res["urls"]:
                st.markdown("<p style='color: var(--cream); font-weight: 500; margin-top: 15px; margin-bottom: 5px;'>Detected Links (Defanged):</p>", unsafe_allow_html=True)
                for url in res["urls"]:
                    st.markdown(f"<span class='link-chip'>{esc(defang(url))}</span>", unsafe_allow_html=True)

            # Deep LLM Text Intelligence Button 
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            llm_disabled = genai_client is None and groq_client is None
            if st.button("✨ Run Deep LLM Contextual Analysis", disabled=llm_disabled):
                with st.spinner("Querying LLM ensemble for threat breakdown..."):
                    prompt = f"You are an expert cybersecurity analyst for 'Beyond Vision', a scam, smishing, and phishing detection engine. Analyze the following text message and any embedded URLs for smishing, social engineering, credential harvesting, or fraudulent intent. Message/Content: \"{st.session_state['last_input']}\" Extracted Links: {res['urls']} Provide your analysis cleanly with: - Intent Verdict - Risk Score (0-100) - Psychological Tactics Used / URL Spoofing Analysis - Concise Threat Summary"
                    
                    report, logs = analyze_text_with_llm(prompt)
                    
                    if report:
                        st.session_state["llm_report"] = clean_markdown(report)
                    else:
                        st.session_state["llm_report"] = f"⚠️ All analytical engines failed.\n\nLogs:\n" + "\n".join(esc(l) for l in logs)

            if st.session_state.get("llm_report"):
                st.markdown("<h3 class='serif-header' style='margin-top: 20px; color: #D4AF37 !important;'>🤖 Deep LLM Intelligence Report</h3>", unsafe_allow_html=True)
                st.info(st.session_state["llm_report"])

# ==========================================
# PAGE 2: CYBER AWARENESS GUIDE (Slider)
# ==========================================
elif nav == "📚 Cyber Awareness Guide":
    st.markdown("<p class='slider-hint'>Swipe or scroll horizontally to explore</p>", unsafe_allow_html=True)

    html_content = (
        '<div class="slider-container">'
        '<div class="dark-plate">'
        '<h3>Cybercrime</h3>'
        '<p>Cybercrime is any crime committed using a computer, phone, or internet network. Instead of a thief breaking into a physical house, a cybercriminal exploits digital avenues to steal credentials, identity, or funds remotely.</p>'
        '</div>'
        '<div class="dark-plate">'
        '<h3>How to protect yourself:</h3>'
        '<p><b>Strong Passwords:</b> Use distinct, complex passphrases mixing letters, numbers, and symbols across accounts.</p>'
        '<p><b>Two-Factor Auth (2FA):</b> Enforce hardware keys or authenticator apps to prevent unauthorized logins.</p>'
        '<p><b>Zero-Trust Links:</b> Never click urgent links in unsolicited SMS or emails claiming account suspension.</p>'
        '<p><b>Patch Promptly:</b> Keep operating systems and browsers updated to fix security vulnerabilities.</p>'
        '</div>'
        '<div class="dark-plate">'
        '<h3>Incident Response</h3>'
        '<p><b>Disconnect:</b> Sever Wi-Fi or cellular connections immediately if unexpected control or malware runs.</p>'
        '<p><b>Rotate Credentials:</b> Change passwords from an uncompromised secondary device immediately.</p>'
        '<p><b>Notify Institutions:</b> Contact financial providers to freeze cards or suspicious transfers.</p>'
        '<p><b>Official Reporting:</b> In India, file complaints via cybercrime.gov.in or call 1930.</p>'
        '</div>'
        '</div>'
    )
    st.markdown(html_content, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
    <div class="footer">
        <div class="footer-title">ENGINEERED BY REHAN</div>
        <a href="https://www.linkedin.com" target="_blank" class="footer-link">
            <span style="color: #0A66C2; font-weight: 600; margin-right: 5px;">in</span> Connect on LinkedIn
        </a>
    </div>
""", unsafe_allow_html=True)
