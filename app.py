import io
import os
import re
import html as html_lib
import streamlit as st
from PIL import Image

# ------------------------------------------------------------------
# Optional-dependency imports
# ------------------------------------------------------------------
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


st.set_page_config(page_title="Beyond Vision", page_icon="🛡️", layout="centered")

# ==========================================
# STYLE — Dark Radial Canvas & Gold Accents
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,500&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --gold: #D4AF37;
        --gold-soft: #E8C766;
        --cream: #EAE3CB;
        --cream-bright: #F7F5EB;
        --muted: #ABABA4;
        --safe: #8BB98F;
        --danger: #C3676B;
        --unknown: #8C9AA6;
        --plate-bg: rgba(255,255,255,0.03);
        --plate-border: rgba(255,255,255,0.08);
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #2A2D24 0%, #0D0E0B 70%) !important;
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer[data-testid="stFooter"], header[data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3, .serif-header {
        font-family: 'Playfair Display', serif !important;
        color: var(--cream) !important;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    p, li, label, span { font-family: 'Inter', sans-serif; }

    ::-webkit-scrollbar { height: 6px; width: 6px; }
    ::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.4); border-radius: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }

    /* Nav dressed up as custom pills */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    div[data-testid="stRadio"] label {
        background: var(--plate-bg);
        border: 1px solid var(--plate-border);
        border-radius: 30px;
        padding: 10px 20px !important;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: rgba(212,175,55,0.5);
    }
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, var(--gold) 0%, #B8912B 100%);
        border-color: var(--gold);
    }
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #16170F !important;
        font-weight: 500;
    }
    div[data-testid="stRadio"] label p {
        color: var(--cream) !important;
        font-size: 0.92rem;
        margin: 0;
    }

    /* Text areas & dropzone */
    .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        color: var(--cream) !important;
        border: 1px solid var(--plate-border) !important;
        border-radius: 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 1px rgba(212,175,55,0.4) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px dashed var(--plate-border) !important;
        border-radius: 16px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(212,175,55,0.12) !important;
        color: var(--cream) !important;
        border: 1px solid rgba(212,175,55,0.3) !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--gold) 0%, #B8912B 100%) !important;
        color: #16170F !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 500 !important;
        padding: 10px 26px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(212,175,55,0.25);
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        border: 1px solid var(--plate-border) !important;
    }
    div[data-testid="stAlertContentError"], div[data-testid="stAlertContentError"] p {
        color: var(--danger) !important;
    }
    div[data-testid="stAlertContentSuccess"], div[data-testid="stAlertContentSuccess"] p {
        color: var(--safe) !important;
    }
    div[data-testid="stAlertContentInfo"], div[data-testid="stAlertContentInfo"] p {
        color: var(--cream) !important;
    }

    [data-testid="stSpinner"] p { color: var(--gold) !important; }

    /* Native container plate */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 24px !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
        padding: 15px !important;
    }

    /* Verdict badge */
    .verdict-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 12px 22px;
        border-radius: 14px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.05rem;
        margin-bottom: 16px;
        border: 1px solid transparent;
    }
    .verdict-spam { background: rgba(195,103,107,0.12); border-color: rgba(195,103,107,0.35); color: var(--danger); }
    .verdict-safe { background: rgba(139,185,143,0.12); border-color: rgba(139,185,143,0.35); color: var(--safe); }
    .verdict-unknown { background: rgba(140,154,166,0.12); border-color: rgba(140,154,166,0.35); color: var(--unknown); }

    /* Confidence meter */
    .confidence-wrap { margin: 4px 0 22px 0; }
    .confidence-track {
        width: 100%;
        height: 8px;
        border-radius: 8px;
        background: rgba(255,255,255,0.06);
        overflow: hidden;
    }
    .confidence-fill { height: 100%; border-radius: 8px; }
    .confidence-caption {
        margin-top: 6px;
        font-size: 0.82rem;
        color: var(--muted);
    }

    /* Detected link chips */
    .link-chip {
        display: inline-block;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--plate-border);
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
    }
    .link-chip a { color: var(--gold); text-decoration: none; }
    .link-chip a:hover { text-decoration: underline; }

    /* Slider Container */
    .slider-container {
        display: flex;
        overflow-x: auto;
        gap: 20px;
        padding-bottom: 20px;
        padding-top: 10px;
        scroll-snap-type: x mandatory;
        -ms-overflow-style: none;
        scrollbar-width: none;
    }
    .slider-container::-webkit-scrollbar { display: none; }

    .dark-plate {
        min-width: min(320px, 88vw);
        max-width: 480px;
        flex: 0 0 85%;
        scroll-snap-align: center;
        background-color: var(--plate-bg);
        border-radius: 24px;
        padding: 35px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid var(--plate-border);
        backdrop-filter: blur(10px);
    }
    .dark-plate h3 {
        margin-top: 0;
        color: var(--gold) !important;
        font-size: 1.5rem;
        margin-bottom: 20px;
    }
    .dark-plate p { font-size: 0.95rem; line-height: 1.7; color: var(--muted); font-weight: 300; }
    .dark-plate b { color: var(--cream); font-weight: 500; }
    .slider-hint {
        text-align: center;
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 1px;
        margin-top: -8px;
        margin-bottom: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 70px;
        padding-top: 30px;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 30px;
    }
    .footer-title {
        font-size: 0.7rem;
        font-weight: 500;
        color: #8C8C87;
        margin-bottom: 15px;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .footer-link {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.03);
        color: var(--cream) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px 28px;
        border-radius: 30px;
        text-decoration: none;
        font-size: 0.9rem;
        transition: 0.3s;
    }
    .footer-link:hover {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    a { color: var(--gold); }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED RESOURCES
# ==========================================
@st.cache_resource(show_spinner=False)
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


def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner=False)
def get_genai_client(api_key):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def run_ocr(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image)


def normalize_verdict(raw_label: str) -> str:
    label = (raw_label or "").strip().upper()
    spam_markers = {"SPAM", "LABEL_1", "1"}
    safe_markers = {"HAM", "NOT SPAM", "SAFE", "LABEL_0", "0"}
    if label in spam_markers:
        return "spam"
    if label in safe_markers:
        return "safe"
    return "unknown"


ai_classifier = load_classifier()
API_KEY = get_api_key()
genai_client = get_genai_client(API_KEY)

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
    "deepfake_report": None,
    "last_media_name": None,
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
        st.info("The text-classification model isn't available right now, so message scanning is offline. Deepfake media scanning below is unaffected.")

    with st.container(border=True):
        st.markdown("<h3 class='serif-header' style='margin-bottom: 15px; font-size: 1.4rem; color: #D4AF37 !important;'>📥 Threat Ingestion Source</h3>", unsafe_allow_html=True)

        input_type = st.radio(
            "Select the type of threat to analyze:",
            ["Text Input (Link/Message)", "Image Upload (Screenshot OCR)", "Media Upload (Deepfake/Voice/Video)"],
            horizontal=False,
        )

        user_input = ""

        if input_type == "Image Upload (Screenshot OCR)":
            uploaded_file = st.file_uploader("Upload a screenshot of the SMS, WhatsApp, or Email", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption="Uploaded Evidence", use_container_width=True)

                if not PYTESSERACT_AVAILABLE:
                    st.error("OCR isn't available on this deployment (pytesseract or tesseract-ocr system package is missing). Paste the message text manually below instead.")
                else:
                    with st.spinner("Extracting text via Optical Character Recognition..."):
                        try:
                            st.session_state["extracted_text"] = run_ocr(image_bytes)
                        except Exception as e:
                            st.session_state["extracted_text"] = ""
                            st.error(f"OCR failed: {e}")

            user_input = st.text_area("Extracted Text (Edit if necessary):", value=st.session_state["extracted_text"], height=150)

        elif input_type == "Media Upload (Deepfake/Voice/Video)":
            uploaded_media = st.file_uploader("Upload Audio (mp3, wav), Video (mp4), or Photo (jpg, png)", type=["png", "jpg", "jpeg", "mp3", "wav", "mp4"])

            if uploaded_media is not None:
                if st.session_state["last_media_name"] != uploaded_media.name:
                    st.session_state["deepfake_report"] = None
                    st.session_state["last_media_name"] = uploaded_media.name

                media_bytes = uploaded_media.getvalue()
                file_name = uploaded_media.name.lower()
                media_type = None

                if file_name.endswith((".png", ".jpg", ".jpeg")):
                    st.image(media_bytes, use_container_width=True)
                    media_type = "image/jpeg" if file_name.endswith((".jpg", ".jpeg")) else "image/png"
                elif file_name.endswith((".mp3", ".wav")):
                    st.audio(media_bytes)
                    media_type = "audio/wav" if file_name.endswith(".wav") else "audio/mp3"
                elif file_name.endswith(".mp4"):
                    st.video(media_bytes)
                    media_type = "video/mp4"

                size_mb = len(media_bytes) / (1024 * 1024)
                if size_mb > 15:
                    st.warning(f"This file is {size_mb:.1f} MB. Large inline uploads can be rejected by the API — trimming it to under ~15 MB is safer.")

                scan_disabled = genai_client is None
                if scan_disabled:
                    st.info("Deepfake scanning needs a Gemini API key. Add GEMINI_API_KEY to .streamlit/secrets.toml or your environment, then reload.")

                if st.button("🔍 Run Deepfake Scan", disabled=scan_disabled):
                    with st.spinner("Analyzing media for AI generation..."):
                        try:
                            media_part = types.Part.from_bytes(data=media_bytes, mime_type=media_type)
                            prompt = "Analyze this media file for indicators of AI generation, synthetic manipulation, or deepfake/voice-cloning artifacts."
                            response = genai_client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=[prompt, media_part],
                            )
                            st.session_state["deepfake_report"] = response.text
                        except Exception as e:
                            st.session_state["deepfake_report"] = None
                            st.error(f"Scan failed: {e}")

            if st.session_state["deepfake_report"]:
                st.markdown("<h3 class='serif-header' style='margin-top: 25px; color: #D4AF37 !important;'>🤖 Forensics Report</h3>", unsafe_allow_html=True)
                st.info(st.session_state["deepfake_report"])

        else:
            user_input = st.text_area("Paste suspicious message or URL here:", height=150)

        if input_type in ["Text Input (Link/Message)", "Image Upload (Screenshot OCR)"]:
            scan_clicked = st.button("Scan Threat Vectors", disabled=(ai_classifier is None))

            if scan_clicked:
                clean_text = user_input.strip()
                if not clean_text:
                    st.warning("Enter or extract some text before scanning.")
                else:
                    with st.spinner("Analyzing threat vectors..."):
                        try:
                            ai_output = ai_classifier(clean_text)[0]
                            st.session_state["scan_result"] = {
                                "verdict": normalize_verdict(ai_output.get("label", "")),
                                "raw_label": ai_output.get("label", "UNKNOWN"),
                                "score": round(ai_output.get("score", 0.0) * 100, 2),
                                "urls": re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', clean_text),
                            }
                        except Exception as e:
                            st.session_state["scan_result"] = {
                                "verdict": "unknown",
                                "raw_label": "ERROR",
                                "score": 0.0,
                                "urls": [],
                                "error": str(e)
                            }

        if st.session_state.get("scan_result"):
            res = st.session_state["scan_result"]
            st.markdown("<h3 class='serif-header' style='margin-top: 25px; color: #D4AF37 !important;'>🧠 Analysis Results</h3>", unsafe_allow_html=True)

            verdict_class = f"verdict-{res['verdict']}"
            verdict_text = "🚨 High Risk / Spam" if res['verdict'] == 'spam' else ("✅ Safe / Clean" if res['verdict'] == 'safe' else "⚠️ Unknown / Inconclusive")

            st.markdown(f"<div class='verdict-badge {verdict_class}'>{verdict_text}</div>", unsafe_allow_html=True)

            st.markdown(f"""
                <div class="confidence-wrap">
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: {res['score']}%; background: {'var(--danger)' if res['verdict'] == 'spam' else 'var(--safe)'};"></div>
                    </div>
                    <div class="confidence-caption">AI Confidence Score: <b>{res['score']}%</b></div>
                </div>
            """, unsafe_allow_html=True)

            if res.get("error"):
                st.error(f"Scanner experienced an error: {res['error']}")

            if res["urls"]:
                st.markdown("<p style='color: var(--cream); font-weight: 500; margin-top: 15px; margin-bottom: 5px;'>Detected Links:</p>", unsafe_allow_html=True)
                for url in res["urls"]:
                    safe_url = html_lib.escape(url)
                    st.markdown(f"<span class='link-chip'><a href='{safe_url}' target='_blank'>{safe_url}</a></span>", unsafe_allow_html=True)


# ==========================================
# PAGE 2: CYBER AWARENESS GUIDE (Slider)
# ==========================================
elif nav == "📚 Cyber Awareness Guide":
    st.markdown("<p class='slider-hint'>Swipe or scroll horizontally to explore</p>", unsafe_allow_html=True)

    # Passing this as a concatenated string prevents Streamlit from EVER rendering it as a code block.
    html_content = (
        '<div class="slider-container">'
        '<div class="dark-plate">'
        '<h3>Cybercrime</h3>'
        '<p>Cybercrime is any crime that is committed using a computer, phone, or the internet. Think of it like digital crime. Instead of a thief breaking into a physical house to steal your wallet, a cybercriminal uses a computer to try and steal your money, your personal information, or your passwords from miles away.</p>'
        '</div>'
        '<div class="dark-plate">'
        '<h3>How to protect yourself:</h3>'
        '<p><b>Make strong passwords:</b> Don\'t use simple words or numbers like "123456." Use a mix of letters, numbers, and symbols, and use a different password for every website.</p>'
        '<p><b>Turn on two-step login (2FA):</b> When you sign into an important account, turn on the option where it texts a code to your phone. That way, even if someone guesses your password, they still can\'t get in.</p>'
        '<p><b>Don\'t click random links:</b> If you get a text or email out of nowhere—even if it looks like it\'s from your bank or a store—saying you need to click a link right away, don\'t do it. It\'s usually a trap.</p>'
        '<p><b>Update your phone and computer:</b> When your device tells you an update is ready, install it. Updates fix hidden security holes that bad guys try to sneak through.</p>'
        '<p><b>Keep your personal info private:</b> Don\'t post your home address, phone number, or birthday publicly on social media.</p>'
        '<p><b>Save your important files:</b> Keep copies of your favorite photos and documents on a backup drive or cloud storage.</p>'
        '</div>'
        '<div class="dark-plate">'
        '<h3>What to do if You are a Victim</h3>'
        '<p><b>Go offline:</b> If your computer or phone is acting crazy, turn off your Wi-Fi or unplug it immediately. This stops the criminal from doing more damage.</p>'
        '<p><b>Change your passwords:</b> Log in (if you still can) and change your passwords right away. Do this for your email, bank, and any other important accounts.</p>'
        '<p><b>Call your bank:</b> If someone stole your money or you gave away your credit card info, call your bank immediately so they can freeze your cards and stop payments.</p>'
        '<p><b>Warn your friends:</b> If a hacker takes over your email or social media, they might send fake messages to your friends. Let everyone know not to click anything sent from your account.</p>'
        '<p><b>Run a virus check:</b> If you have security software on your device, run a full scan to clean out any bad files or viruses.</p>'
        '<p><b>Report it:</b> Tell the authorities. In India, you can report cybercrimes online at cybercrime.gov.in or call the helpline number 1930.</p>'
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
