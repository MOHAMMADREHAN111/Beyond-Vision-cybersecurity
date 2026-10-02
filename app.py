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
    # Static fallbacks: when discovery worked, keep only models that really exist.
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
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            errors.append(f"{model}: {str(exc)[:220]}")
            if _is_auth_error(exc):
                break  # no point trying other models with a bad key
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
        except Exception as exc:  # noqa: BLE001
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
# Whole-word matching: the old substring check made "pan" fire on "company", "won" on "wonderful".
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
        flags.append("Triggered high-risk social-engineering keywords: " + esc(", ".join(matched)) + ".")
