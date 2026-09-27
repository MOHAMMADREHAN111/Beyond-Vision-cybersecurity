import io
import os
import re
import html as html_lib
from urllib.parse import urlparse
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

st.set_page_config(page_title="Beyond Vision", page_icon="☁️", layout="centered")

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
        --warning: #E0A96D;
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
        flex-direction: column;
        align-items: flex-start;
        padding: 14px 22px;
        border-radius: 14px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        margin-bottom: 16px;
        border: 1px solid transparent;
        width: 100%;
    }
    .verdict-spam { background: rgba(195,103,107,0.12); border-color: rgba(195,103,107,0.35); color: var(--danger); }
    .verdict-safe { background: rgba(139,185,143,0.12); border-color: rgba(139,185,143,0.35); color: var(--safe); }
    .verdict-unknown { background: rgba(140,154,166,0.12); border-color: rgba(140,154,166,0.35); color: var(--unknown); }

    /* Confidence meter */
    .confidence-wrap { margin: 4px 0 20px 0; }
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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# RESOURCE INITIALIZATION
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
    "llm_report": None,
    "deepfake_report": None,
    "last_media_name": None,
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
        st.info("The text-classification model is offline. Deepfake media scanning below is unaffected.")

    with st.container(border=True):
        st.markdown("<h3 class='serif-header' style='margin-bottom: 15px; font-size: 1.4rem; color: #D4AF37 !important;'>📥 Threat Ingestion Source</h3>", unsafe_allow_html=True)

        input_type = st.radio(
            "Select the type of threat to analyze:",
            ["Text Input (Link/Message)", "Image Upload (Screenshot OCR)", "Media Upload (Deepfake/Voice/Video)"],
            horizontal=False,
        )

        user_input = ""

        # Option A: OCR Upload
        if input_type == "Image Upload (Screenshot OCR)":
            uploaded_file = st.file_uploader("Upload a screenshot of the SMS, WhatsApp, or Email", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption="Uploaded Evidence", use_container_width=True)

                if not PYTESSERACT_AVAILABLE:
                    st.error("OCR is unavailable on this server. Please paste the text manually below.")
                else:
                    with st.spinner("Extracting text via Optical Character Recognition..."):
                        try:
                            st.session_state["extracted_text"] = run_ocr(image_bytes)
                        except Exception as e:
                            st.session_state["extracted_text"] = ""
                            st.error(f"OCR failed: {e}")

            user_input = st.text_area("Extracted Text (Edit if necessary):", value=st.session_state["extracted_text"], height=150)

        # Option B: Deepfake Forensics Upload
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
                    st.warning(f"File size is {size_mb:.1f} MB. Inline files over 15 MB may be rejected by the API.")

                scan_disabled = genai_client is None
                if scan_disabled:
                    st.info("Deepfake scanning requires a Gemini API key. Add GEMINI_API_KEY in Streamlit Secrets.")

                if st.button("🔍 Run Deepfake Scan", disabled=scan_disabled):
                    with st.spinner("Analyzing media for AI generation artifacts..."):
                        try:
                            media_part = types.Part.from_bytes(data=media_bytes, mime_type=media_type)
                            prompt = """
                            You are a senior forensic analyst for 'Beyond Vision'. Analyze this media file for indicators of AI generation, synthetic manipulation, or deepfake/voice-cloning artifacts.
                            Provide:
                            1. Synthetic Probability Verdict (Real vs. AI-Generated)
                            2. Confidence Score (0-100%)
                            3. Key Forensic Observations
                            4. Potential Scam Context
                            """
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

        # Option C: Direct Text Ingestion
        else:
            user_input = st.text_area("Paste suspicious message or URL here:", value=st.session_state.get("last_input", ""), height=150)

        # Core Text/OCR Execution Logic
        if input_type in ["Text Input (Link/Message)", "Image Upload (Screenshot OCR)"]:
            scan_clicked = st.button("Scan Threat Vectors", disabled=(ai_classifier is None))

            if scan_clicked:
                clean_text = user_input.strip()
                if not clean_text:
                    st.warning("Enter or extract some text before scanning.")
                else:
                    with st.spinner("Analyzing threat vectors..."):
                        st.session_state["last_input"] = clean_text
                        st.session_state["llm_report"] = None

                        # 1. ML Classifier
                        try:
                            ai_output = ai_classifier(clean_text)[0]
                            ai_label = ai_output.get("label", "UNKNOWN").upper()
                            ai_score = round(ai_output.get("score", 0.0) * 100, 2)
                        except Exception:
                            ai_label = "ERROR"
                            ai_score = 0.0

                        # 2. Original Heuristic & TLD Engine
                        risk_score = 15
                        heuristics = []
                        lower_text = clean_text.lower()

                        keywords = ["urgent", "locked", "verify", "suspended", "won", "prize", "tax", "pan", "bonus", "deposit", "kyc", "wa.link", "claim", "lottery"]
                        matched_words = [w for w in keywords if w in lower_text]
                        if matched_words:
                            risk_score += 35
                            heuristics.append(f"Triggered high-risk social engineering keywords: {', '.join(matched_words)}.")

                        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', clean_text)
                        for url in urls:
                            normalized_url = url if url.startswith("http") else f"http://{url}"
                            domain = urlparse(normalized_url).netloc.lower()

                            suspicious_tlds = [".xyz", ".top", ".buzz", ".cc", ".ru", ".fit", ".tk", ".ml"]
                            if any(domain.endswith(tld) for tld in suspicious_tlds):
                                risk_score += 35
                                heuristics.append(f"Suspicious high-risk TLD detected: {domain}")

                            if re.match(r'^\d{1,3}(\.\d{1,3}){3}', domain):
                                risk_score += 40
                                heuristics.append(f"Direct numeric IP address link detected: {domain}")

                            brands = ["paytm", "sbi", "icici", "hdfc", "netflix", "amazon", "google", "binance", "whatsapp"]
                            if any(b in domain for b in brands) and not domain.endswith((".gov.in", ".co.in", ".com")):
                                risk_score += 35
                                heuristics.append(f"Potential brand spoofing detected in domain: {domain}")

                        if urls and not heuristics:
                            heuristics.append("Contains embedded links or domains.")
                        elif not heuristics:
                            heuristics.append("No overt structural heuristic red flags detected.")

                        if ai_label in ["SPAM", "LABEL_1", "1"]:
                            risk_score = max(risk_score, int(ai_score * 0.85))

                        risk_score = min(risk_score, 100)

                        st.session_state["scan_result"] = {
                            "ai_label": ai_label,
                            "ai_score": ai_score,
                            "risk_score": risk_score,
                            "heuristics": heuristics,
                            "urls": urls,
                        }

        # Results Display
        if st.session_state.get("scan_result"):
            res = st.session_state["scan_result"]

            st.markdown("<h3 class='serif-header' style='margin-top: 25px; color: #D4AF37 !important;'>🧠 Message & Content Classifier</h3>", unsafe_allow_html=True)

            is_malicious = res["risk_score"] >= 50 or res["ai_label"] in ["SPAM", "LABEL_1", "1"]
            verdict_badge_class = "verdict-spam" if is_malicious else "verdict-safe"
            verdict_title = "MALICIOUS / FRAUDULENT" if is_malicious else "SAFE / CLEAN"

            st.markdown(f"""
                <div class='verdict-badge {verdict_badge_class}'>
                    <span style='font-size: 1.15rem; font-weight: 600;'>Threat Verdict: {verdict_title}</span>
                    <span style='font-size: 0.88rem; opacity: 0.85;'>(Combined Risk Score: {res['risk_score']}/100)</span>
                </div>
            """, unsafe_allow_html=True)

            # Confidence Track
            st.markdown(f"""
                <div class="confidence-wrap">
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: {res['risk_score']}%; background: {'var(--danger)' if is_malicious else 'var(--safe)'};"></div>
                    </div>
                    <div class="confidence-caption">Risk Index: <b>{res['risk_score']}%</b> | Model Confidence: <b>{res['ai_score']}%</b></div>
                </div>
            """, unsafe_allow_html=True)

            # Heuristics Section
            st.markdown("<h3 class='serif-header' style='margin-top: 15px; font-size: 1.25rem; color: #EAE3CB !important;'>⚙️ Heuristic Breakdown</h3>", unsafe_allow_html=True)
            for item in res.get("heuristics", []):
                st.markdown(f"<p style='color: var(--muted); font-size: 0.92rem; margin: 4px 0;'>• {item}</p>", unsafe_allow_html=True)

            if res["urls"]:
                st.markdown("<p style='color: var(--cream); font-weight: 500; margin-top: 15px; margin-bottom: 5px;'>Detected Links:</p>", unsafe_allow_html=True)
                for url in res["urls"]:
                    safe_url = html_lib.escape(url)
                    st.markdown(f"<span class='link-chip'><a href='{safe_url}' target='_blank'>{safe_url}</a></span>", unsafe_allow_html=True)

            # Deep LLM Text Intelligence Button (Restored AI Feature)
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            llm_disabled = genai_client is None
            if st.button("✨ Run Deep LLM Contextual Analysis", disabled=llm_disabled):
                with st.spinner("Querying Gemini AI for threat breakdown..."):
                    try:
                        prompt = f"""
                        You are an expert cybersecurity analyst for 'Beyond Vision', a scam, smishing, and phishing detection engine.
                        Analyze the following text message and any embedded URLs for smishing, social engineering, credential harvesting, or fraudulent intent.

                        Message/Content: "{st.session_state['last_input']}"
                        Extracted Links: {res['urls']}

                        Provide your analysis cleanly with:
                        - Intent Verdict
                        - Risk Score (0-100)
                        - Psychological Tactics Used / URL Spoofing Analysis
                        - Concise Threat Summary
                        """
                        response = genai_client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt,
                        )
                        st.session_state["llm_report"] = response.text
                    except Exception as e:
                        st.session_state["llm_report"] = f"⚠️ Gemini connection error: {str(e)}"

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
