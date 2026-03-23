import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

st.set_page_config(page_title="AI Text Summarizer", layout="wide", page_icon="✦")

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Mono:wght@300;400&family=Lato:wght@300;400&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0d0d;
    color: #e8e0d5;
    font-family: 'Lato', sans-serif;
    font-weight: 300;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 10% 20%, rgba(212,175,100,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(180,120,60,0.04) 0%, transparent 50%),
        #0d0d0d;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

.block-container {
    max-width: 820px !important;
    padding: 2.5rem 2rem 4rem !important;
    margin: 0 auto;
}

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid rgba(212,175,100,0.12);
    margin-bottom: 2.5rem;
}
.hero-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    color: #c9a84c;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    opacity: 0.75;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.6rem, 5vw, 4rem);
    font-weight: 700;
    color: #f0e6d3;
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin: 0 0 0.8rem;
}
.hero-title span { color: #c9a84c; }
.hero-sub {
    font-size: 0.92rem;
    color: #6a6050;
    letter-spacing: 0.02em;
}

/* Section labels */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.28em;
    color: #c9a84c;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.65;
}

/* Radio */
[data-testid="stRadio"] label,
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important;
    color: #a09080 !important;
    letter-spacing: 0.04em;
}

/* Textarea */
textarea {
    background: #111111 !important;
    border: 1px solid rgba(212,175,100,0.14) !important;
    border-radius: 2px !important;
    color: #d4c9b8 !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 300 !important;
    font-size: 0.92rem !important;
    line-height: 1.75 !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
textarea:focus {
    border-color: rgba(201,168,76,0.38) !important;
    box-shadow: none !important;
    outline: none !important;
}
textarea::placeholder { color: #3a3428 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111111;
    border: 1px dashed rgba(212,175,100,0.18);
    border-radius: 2px;
    padding: 1.2rem 1.5rem;
}
[data-testid="stFileUploader"] label {
    color: #6a6050 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #111111 !important;
    border: 1px solid rgba(212,175,100,0.14) !important;
    border-radius: 2px !important;
    color: #d4c9b8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Primary button */
[data-testid="stButton"] > button {
    background: #c9a84c !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.72rem 2.5rem !important;
    font-weight: 400 !important;
    width: 100%;
    margin-top: 0.4rem;
    transition: background 0.18s, transform 0.1s !important;
}
[data-testid="stButton"] > button:hover {
    background: #dfc060 !important;
    transform: translateY(-1px) !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #c9a84c !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.18s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(201,168,76,0.07) !important;
    border-color: #c9a84c !important;
}

/* Summary card */
.summary-card {
    background: #0f0f0f;
    border: 1px solid rgba(212,175,100,0.15);
    border-left: 3px solid #c9a84c;
    border-radius: 2px;
    padding: 1.6rem 2rem;
    margin-top: 0.8rem;
    font-family: 'Lato', sans-serif;
    font-weight: 300;
    font-size: 1.05rem;
    line-height: 1.9;
    color: #d8cfc2;
    letter-spacing: 0.01em;
}
.summary-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.32em;
    color: #c9a84c;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    opacity: 0.75;
}

/* Alerts */
[data-testid="stAlert"] {
    background: rgba(201,168,76,0.05) !important;
    border: 1px solid rgba(201,168,76,0.18) !important;
    border-radius: 2px !important;
    color: #a08840 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stSuccess"] {
    background: rgba(100,175,120,0.05) !important;
    border: 1px solid rgba(100,175,120,0.18) !important;
    border-radius: 2px !important;
    color: #72a882 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0f0f0f !important;
    border: 1px solid rgba(212,175,100,0.1) !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #6a6050 !important;
    letter-spacing: 0.05em;
}

/* Spinner */
[data-testid="stSpinner"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #6a6050 !important;
    letter-spacing: 0.12em;
}

/* Input labels */
label[data-testid="stWidgetLabel"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color: #5a5040 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #2e2818; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #c9a84c; }
</style>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ─── Helpers ─────────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif file_type == "pdf":
        try:
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        except ImportError:
            st.error("pdfplumber not installed. Run: pip install pdfplumber")
            return ""

    elif file_type == "docx":
        try:
            import docx
            from io import BytesIO
            doc = docx.Document(BytesIO(uploaded_file.read()))
            return "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            st.error("python-docx not installed. Run: pip install python-docx")
            return ""

    else:
        st.error(f"Unsupported file type: .{file_type}")
        return ""

CHUNK_TOKEN_LIMIT = 900

def split_into_chunks(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk, current_len = [], [], 0
    for sentence in sentences:
        token_len = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_len + token_len > CHUNK_TOKEN_LIMIT and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [sentence], token_len
        else:
            current_chunk.append(sentence)
            current_len += token_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_chunk(text, max_length, min_length):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    n_tokens = inputs["input_ids"].shape[1]
    safe_min = min(min_length, max(1, n_tokens // 2))
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=safe_min,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_text(text, length="Balanced", progress_callback=None):
    length_params = {
        "Short":    {"max_length": 60,  "min_length": 20},
        "Balanced": {"max_length": 120, "min_length": 40},
        "Detailed": {"max_length": 200, "min_length": 80},
    }
    params = length_params[length]
    chunks = split_into_chunks(text)
    total = len(chunks)

    # Pass 1: summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, total, f"Summarizing chunk {i+1} of {total}...")
        chunk_max = max(40, params["max_length"] // max(1, total // 3 + 1))
        chunk_min = max(10, chunk_max // 3)
        chunk_summaries.append(summarize_chunk(chunk, chunk_max, chunk_min))

    combined = " ".join(chunk_summaries)

    # Pass 2: compress if still too long
    combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
    if combined_tokens > CHUNK_TOKEN_LIMIT:
        if progress_callback:
            progress_callback(total, total, "Running final compression pass...")
        pass2 = [summarize_chunk(c, params["max_length"], params["min_length"] // 2)
                 for c in split_into_chunks(combined)]
        combined = " ".join(pass2)

    # Pass 3: final polish
    if progress_callback:
        progress_callback(total, total, "Polishing final summary...")
    if len(tokenizer.encode(combined, add_special_tokens=False)) > 50:
        combined = summarize_chunk(combined, params["max_length"], params["min_length"])

    return combined, total

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ AI-Powered &nbsp;·&nbsp; NLP &nbsp;·&nbsp; AI Text Summarizer</div>
    <h1 class="hero-title">AI <span>Text</span> Summarizer</h1>
    <p class="hero-sub">Paste text or upload a document — get the essence in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ─── Input Mode ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Input Source</div>', unsafe_allow_html=True)
input_mode = st.radio(
    "",
    ["✍️  Paste Text", "📁  Upload File"],
    horizontal=True,
    label_visibility="collapsed"
)

text_to_summarize = ""
st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

if input_mode == "✍️  Paste Text":
    input_text = st.text_area(
        "Text Input",
        height=230,
        placeholder="Start typing or paste your content here…",
        label_visibility="collapsed"
    )
    text_to_summarize = input_text

else:
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        with st.spinner("Extracting text…"):
            extracted = extract_text_from_file(uploaded_file)

        if extracted.strip():
            word_count = len(extracted.split())
            st.success(f"✓  {word_count:,} words extracted from {uploaded_file.name}")
            with st.expander("Preview extracted text"):
                st.write(extracted[:2000] + ("…" if len(extracted) > 2000 else ""))
            text_to_summarize = extracted
        else:
            st.warning("Could not extract text. The file may be empty or image-based.")

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ─── Controls ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Summary Length</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    summary_length = st.selectbox("", ["Short", "Balanced", "Detailed"], label_visibility="collapsed")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

if st.button("Summarize  →"):
    if not text_to_summarize.strip():
        st.warning("Please provide some text or upload a file first.")
    else:
        cleaned = clean_text(text_to_summarize)
        word_count = len(cleaned.split())
        chunks_preview = split_into_chunks(cleaned)
        n_chunks = len(chunks_preview)

        status_text = st.empty()
        progress_bar = st.empty()

        def update_progress(i, total, message):
            pct = int((i / max(total, 1)) * 100)
            status_text.markdown(
                f"<div style='font-family:DM Mono,monospace;font-size:0.72rem;"
                f"color:#6a6050;letter-spacing:0.1em;margin-bottom:0.4rem;'>{message}</div>",
                unsafe_allow_html=True
            )
            progress_bar.progress(pct)

        if n_chunks > 1:
            st.info(f"📄  Long document detected — {word_count:,} words split into {n_chunks} chunks for full coverage.")

        summary, total_chunks = summarize_text(cleaned, length=summary_length, progress_callback=update_progress)

        status_text.empty()
        progress_bar.empty()

        chunk_note = f"{total_chunks} chunk{'s' if total_chunks != 1 else ''}" if total_chunks > 1 else "single pass"

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-header">✦ Summary &nbsp;·&nbsp; {summary_length} &nbsp;·&nbsp; {chunk_note}</div>
            {summary}
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="⬇  Download Summary",
            data=summary,
            file_name="ai_text_summarizer_summary.txt",
            mime="text/plain"
        )